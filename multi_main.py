import os
import math
import time
import logging
import threading
import queue
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import torch

from camera.CameraThread import CameraThread
from detectors.PersonDetector import PersonDetector
from detectors.motion_detector import MotionDetector
from trackers.person_tracker import PersonTracker
from recognition.person_db_recognizer import PersonDBRecognizer
from recognition.unknown_gallery import UnknownGallery
from trackers.zone_counter import ZoneCounter
from database import task_db

# ===============================
# FIX SEGFAULT: lock toàn cục cho MỌI lần chạy inference model native
# (PyTorch/YOLO lẫn ONNXRuntime/insightface)
# ===============================
# Nguyên nhân segfault thực tế: mỗi camera chạy 1 CameraPipeline riêng
# (thread riêng), MỖI pipeline có 1 model YOLO (PyTorch) RIÊNG - và
# RecognitionWorker (cũng 1 thread riêng) chạy insightface (ONNXRuntime).
# Khi >= 2 trong số các model này CÙNG LÚC chạy forward() (vd 2 camera cùng
# có người đi vào 1 lúc), PyTorch/ONNXRuntime tranh chấp CHUNG 1 threadpool
# nội bộ (OpenMP/MKL) ở cấp TIẾN TRÌNH - dù là model instance khác nhau,
# việc gọi đồng thời từ nhiều thread Python KHÔNG được đảm bảo an toàn ở
# tầng C++ -> lỗi bộ nhớ -> Segmentation fault (không bắt được bằng
# try/except vì đây là crash native, không phải exception Python).
#
# Lock này ép TOÀN BỘ inference (mọi camera + nhận diện khuôn mặt) chạy
# TUẦN TỰ, không bao giờ đồng thời - đổi lại 1 chút thông lượng (throughput)
# để đổi lấy việc không còn crash. Đây là cách khắc phục triệt để nhất mà
# không cần đổi kiến trúc sang multiprocessing (mỗi camera 1 tiến trình
# riêng - cũng loại bỏ được lỗi này nhưng tốn công sửa nhiều hơn nhiều).
ML_INFERENCE_LOCK = threading.Lock()

# Giảm số luồng nội bộ mà torch tự spawn (mặc định torch dùng NHIỀU luồng
# OpenMP cho 1 lần forward - cộng với việc chạy nhiều model song song ở
# nhiều thread Python càng làm tăng nguy cơ tranh chấp threadpool nói trên).
# Đặt = 1 giúp giảm đáng kể bề mặt xảy ra xung đột, dùng kèm với
# ML_INFERENCE_LOCK ở trên (2 lớp phòng thủ, không thay thế nhau).
torch.set_num_threads(1)

# ===============================
# LOGGING
# ===============================
# Chạy nền dài ngày thì print() ra console không đủ - console có thể không
# ai xem, hoặc bị đóng cùng terminal. Ghi thêm ra file xoay vòng (rotating,
# tối đa 5 file x 10MB) để sau này debug sự cố vẫn còn log tra lại được.
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # vẫn in ra console như trước
        RotatingFileHandler(
            os.path.join(LOG_DIR, "multi_main.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("multi_main")

# Gọi định kỳ để ghi bù các face_event từng bị lỗi khi ghi Mongo (xem
# task_db.py - _buffer_face_event_to_disk / flush_pending_face_events)
PENDING_FLUSH_INTERVAL = 90  # giây

# ===============================
# CONFIG
# ===============================

MODEL_PATH = os.path.join("detectors", "yolov8n.pt")
CONFIDENCE = 0.5
YOLO_IMGSZ = 640  # giảm từ 640 mặc định -> YOLO chạy nhanh hơn nhiều trên CPU

# Giới hạn tổng số luồng CPU mà PyTorch/YOLO được dùng (áp dụng CHO CẢ TIẾN
# TRÌNH, không phải riêng từng camera). Để trống 1-2 lõi cho việc đọc camera
# và hiển thị không bị đói CPU. Thử nghiệm và chỉnh theo số lõi CPU thật của máy.
TORCH_NUM_THREADS = 4

# LƯU Ý QUAN TRỌNG: KHÔNG bỏ bớt tần suất gọi model.track() một cách rời rạc
# (kiểu "mỗi N frame") khi đang có chuyển động - ByteTrack (persist=True) cần
# được gọi ĐỀU ĐẶN, liên tục để giữ track. Gọi thưa/không đều -> ByteTrack mất
# dấu liên tục -> sinh ID mới hoài (VD thấy ID nhảy 1 -> 5 bỏ qua 2,3,4).
# Thay vào đó, dùng "motion hold" bên dưới để tránh Motion Detector báo có/
# không chuyển động chập chờn (cũng gây hiệu ứng tương tự).
MOTION_HOLD_FRAMES = 5  # sau khi hết chuyển động thật, vẫn coi là "có chuyển động"
                         # thêm N frame nữa trước khi tắt hẳn YOLO - chống nhiễu

SAVE_DIR = "saved_person"
EXIT_TIMEOUT = 30

ENABLE_MOTION_GATE = True
MOTION_MIN_AREA = 5000
MOTION_HISTORY = 500
MOTION_VAR_THRESHOLD = 16

TRACKER_TOP_K = 3

ENABLE_FACE_ID = True
FACE_CTX_ID = -1          # -1 = CPU, 0 = GPU đầu tiên
FACE_THRESHOLD = 0.45     # ngưỡng cosine similarity - cần tự test/tinh chỉnh
FACE_REFRESH_INTERVAL = 60  # giây - tự load lại persons từ DB sau mỗi khoảng này


ENABLE_DB_LOGGING = True

# -------------------------------------------------------------------------
# ĐÃ BỎ CƠ CHẾ DEDUP THEO THỜI GIAN (cooldown) VÀ ĐỐI CHIẾU THEO ĐỊA ĐIỂM
# (cross-camera resolve): theo yêu cầu nghiệp vụ mới, HỄ THẤY MẶT LÀ GHI
# NGAY vào DB - không còn giữ lại (treo) để chờ so sánh giữa các camera,
# cũng không còn chặn ghi lặp theo khoảng cách thời gian giữa 2 lần thấy.
# RecognitionWorker giờ chỉ còn: nhận diện xong track nào -> có mặt thì ghi
# DB ngay track đó, xong.
#
# LƯU Ý: việc ĐẾM SỐ LƯỢNG (đang lưu trú, vào/ra...) không dựa vào chuyện
# DB có bị ghi trùng nhiều dòng hay không - toàn bộ được tính ở TẦNG
# DATABASE (task_db.get_current_staying() / get_current_staying_count()),
# các hàm này luôn lấy event MỚI NHẤT theo (identity, address) nên có ghi
# trùng nhiều dòng liên tiếp cũng KHÔNG làm sai số đếm, chỉ tốn thêm bản
# ghi lịch sử/ảnh trong face_events.

# -------------------------------------------------------------------------
# NGƯỜI LẠ (chưa enroll) - NHỚ LẠI khi họ quay lại sau X phút/giờ.
#
# Trước đây: mọi người lạ đều bị ghi person_id=-1, KHÔNG có cách nào biết
# "người lạ này" và "người lạ lúc nãy" là 1 người - mỗi lần xuất hiện lại
# đều bị coi là hoàn toàn mới. UnknownGallery (recognition/unknown_gallery.py)
# giữ 1 "bộ nhớ tạm" (chỉ trong RAM) các embedding người lạ gần đây, cấp
# 1 temp_id ổn định cho mỗi người - dùng để dedup theo hướng + đối chiếu
# chéo camera (xem bên dưới), KHÔNG đổi schema Mongo (vẫn lưu person_id=-1).
#
# UNKNOWN_GALLERY_TTL_SECONDS: người lạ không quay lại trong khoảng này sẽ
# bị dọn khỏi bộ nhớ (tránh phình RAM vô hạn). Đặt DÀI hơn nhiều so với
# "10 phút" trong yêu cầu ban đầu để có biên an toàn.
UNKNOWN_GALLERY_THRESHOLD = FACE_THRESHOLD
UNKNOWN_GALLERY_TTL_SECONDS = 6 * 3600  # 6 tiếng

# Hiển thị dạng lưới - kích thước mỗi ô (mỗi camera) trong lưới.
# TRƯỚC ĐÂY: 480x270 - quá nhỏ so với màn hình hiển thị thực tế -> khi
# cửa sổ bị phóng to/fullscreen, ảnh (và cả box detection vẽ trên đó) bị
# kéo giãn nhiều lần -> vỡ khối (blocky), trông như "lỗi ảnh" dù frame gốc
# từ camera hoàn toàn nét. Tăng lên gần với độ phân giải hiển thị thật
# (VD màn hình 1920x1080, chia lưới 2x2 thì mỗi ô nên ~960x540) để giảm
# hẳn hiện tượng này. Chỉnh số này theo đúng số camera + độ phân giải
# màn hình đang dùng.
GRID_CELL_WIDTH = 960
GRID_CELL_HEIGHT = 540

WINDOW_NAME = "Multi-Camera Grid"
FULLSCREEN_ON_START = False


def _normalize_url(url):
    # url lưu trong Mongo luôn là string - nếu là số (vd "0" cho webcam
    # local) thì convert sang int cho cv2.VideoCapture, còn RTSP thì giữ
    # nguyên string.
    if isinstance(url, str) and url.strip().lstrip("-").isdigit():
        return int(url)
    return url


def load_identity_cameras_from_db():
    """
    Camera "cố định 1 chiều" (như cũ) - dùng cho nhận diện danh tính, ghi
    face_events + person_sessions theo is_in cố định của camera.

    Đọc camera đang bật (status=True) VÀ camera_role KHÁC "occupancy" (bao
    gồm camera_role="identity" hoặc camera CŨ chưa từng có field này - xem
    task_db.get_identity_cameras()). Trả về format main() cần:
    [{"name": <channel>, "url": <rtsp_url hoặc int>}].

    "name" lấy từ field "channel" trong DB - PHẢI khớp với channel dùng khi
    ghi face_events (task_db.save_face_to_db), để tính năng vào/ra hoạt động
    đúng (xem task_db.get_current_staying()/get_person_movements()).
    """
    docs = task_db.get_identity_cameras()

    cams = []

    for d in docs:
        url = d.get("url")
        channel = d.get("channel")

        if not channel:
            logger.warning(f"Bỏ qua camera thiếu channel (_id={d.get('_id')})")
            continue

        if url is None or url == "":
            logger.warning(f"Camera '{channel}' chưa có url trong DB, bỏ qua "
                  f"(sửa bằng: python manage_cameras.py edit --channel {channel} --url ...)")
            continue

        cams.append({"name": channel, "url": _normalize_url(url)})

    return cams


def load_occupancy_cameras_from_db():
    """
    Camera "thấy cả 2 chiều" (mới) - CHỈ đếm số người ra/vào ẨN DANH bằng 2
    vùng polygon (ZoneCounter), KHÔNG chạy face recognition. Đọc camera
    đang bật (status=True) VÀ camera_role="occupancy" (xem
    task_db.get_occupancy_cameras()/configure_occupancy_camera()).

    Camera nào chưa được cấu hình zone_in/zone_out (chưa chạy
    configure_zone_camera.py) sẽ bị BỎ QUA + log cảnh báo, thay vì crash.
    """
    docs = task_db.get_occupancy_cameras()

    cams = []

    for d in docs:
        url = d.get("url")
        channel = d.get("channel")
        zone_in = d.get("zone_in")
        zone_out = d.get("zone_out")

        if not channel:
            logger.warning(f"Bỏ qua occupancy camera thiếu channel (_id={d.get('_id')})")
            continue

        if url is None or url == "":
            logger.warning(f"Occupancy camera '{channel}' chưa có url trong DB, bỏ qua")
            continue

        if not zone_in or not zone_out:
            logger.warning(
                f"Occupancy camera '{channel}' chưa được cấu hình zone_in/zone_out - "
                f"chạy: python configure_zone_camera.py --channel {channel}"
            )
            continue

        cams.append({
            "name": channel,
            "url": _normalize_url(url),
            "zone_in": zone_in,
            "zone_out": zone_out,
            # address của cơ sở - dùng để load số người HIỆN ĐANG có mặt từ
            # task_db.get_zone_occupancy_count() ngay lúc khởi động (xem
            # ZoneCameraPipeline.__init__), thay vì luôn bắt đầu từ 0.
            "address": d.get("address"),
        })

    return cams


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        track_id = det["track_id"]
        conf = det["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{track_id} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )


def build_grid(output_frames, cam_names, cell_width=GRID_CELL_WIDTH, cell_height=GRID_CELL_HEIGHT):
    """
    Ghép frame mới nhất của từng camera thành 1 ảnh lưới duy nhất.
    Tự tính số hàng/cột dựa theo số lượng camera (gần vuông nhất có thể).
    Camera nào chưa có frame (mới khởi động / mất kết nối) sẽ hiện ô "no signal".
    """
    n = len(cam_names)
    if n == 0:
        return None

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    canvas = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)

    for idx, name in enumerate(cam_names):
        r = idx // cols
        c = idx % cols

        frame = output_frames.get(name)

        if frame is None:
            cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            cv2.putText(
                cell,
                f"{name}: no signal",
                (20, cell_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            # INTER_AREA cho kết quả downscale nét hơn, ít răng cưa hơn so
            # với INTER_LINEAR mặc định khi thu nhỏ ảnh (đặc biệt ảnh có
            # nhiều chi tiết/nhiễu hạt như camera thiếu sáng) - INTER_LINEAR
            # chỉ tối ưu khi phóng to, không phải khi thu nhỏ.
            cell = cv2.resize(
                frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA
            )

        y1 = r * cell_height
        y2 = y1 + cell_height
        x1 = c * cell_width
        x2 = x1 + cell_width

        canvas[y1:y2, x1:x2] = cell

    return canvas


class CameraPipeline(threading.Thread):
    """
    Mỗi camera chạy trong 1 thread riêng, với CameraThread + PersonDetector
    (YOLO model riêng) + PersonTracker (trạng thái track riêng).

    Chỉ ghi frame kết quả vào `output_frames[name]` - KHÔNG gọi cv2.imshow
    ở đây, để đảm bảo toàn bộ thao tác GUI chỉ chạy ở thread chính.

    LƯU Ý: PersonTracker ở đây KHÔNG dùng face_embedder (insightface) - vì
    insightface là tác vụ CPU nặng và PHẢI được chạy trong RecognitionWorker
    (thread nền riêng), không phải trong thread đọc camera này. Việc merge
    track bị đứt gãy ngắn hạn (vài giây) vẫn dùng đặc trưng ngoại hình
    (histogram màu) như mặc định của PersonTracker - đủ nhẹ để chạy mỗi
    frame mà không ảnh hưởng tốc độ đọc camera.
    """

    def __init__(
        self,
        name,
        camera_url,
        output_frames,
        stop_event,
        recognition_queue,
    ):
        super().__init__(daemon=True, name=f"Pipeline-{name}")

        self.cam_name = name
        self.camera = CameraThread(camera_url)
        self.detector = PersonDetector(
            model_name=MODEL_PATH,
            confidence=CONFIDENCE,
            imgsz=YOLO_IMGSZ,
        )

        if ENABLE_MOTION_GATE:
            # cooldown=0 -> không throttle, kiểm tra chuyển động MỖI frame
            # (bản gốc dùng cooldown để chống báo động lặp lại, không phù hợp
            # để làm cổng lọc chạy liên tục cho tracking)
            self.motion_detector = MotionDetector(
                min_area=MOTION_MIN_AREA,
                cooldown=0,
                history=MOTION_HISTORY,
                var_threshold=MOTION_VAR_THRESHOLD,
            )
        else:
            self.motion_detector = None

        self.recognition_queue = recognition_queue

        self.output_frames = output_frames
        self.stop_event = stop_event
        self.frame_index = 0
        self._last_motion_frame = -999  # frame gần nhất thực sự có chuyển động

        self.tracker = PersonTracker(
            save_dir=os.path.join(SAVE_DIR, name),
            top_k=TRACKER_TOP_K,
            exit_timeout=EXIT_TIMEOUT,
            on_person_saved=self._on_person_saved,
        )

    def _on_person_saved(self, track_id, crops, disappear_time):
        # CHỈ đẩy vào hàng đợi - không gọi insightface ở đây, để không chặn
        # (block) luồng đọc/xử lý camera này. Việc nhận diện + ghi DB thật sự
        # được xử lý ở RecognitionWorker (thread nền riêng, xem bên dưới).
        # disappear_time không còn dùng để đối chiếu camera đối diện nữa
        # (đã bỏ cơ chế đó) - vẫn giữ trong tuple để không phải đổi chữ ký
        # hàng đợi, RecognitionWorker hiện chỉ bỏ qua giá trị này.
        self.recognition_queue.put((self.cam_name, track_id, crops, disappear_time))

    def run(self):
        if not self.camera.start():
            # QUAN TRỌNG: KHÔNG return ở đây nữa. Trước đây nếu camera lỗi
            # ngay từ lần thử đầu tiên (vd NVR đang khởi động lại đúng lúc
            # multi_main.py start), toàn bộ pipeline camera này sẽ thoát
            # hẳn và KHÔNG BAO GIỜ được thử lại trong suốt phiên chạy.
            #
            # Giờ CameraThread (xem camera/CameraThread.py) tự chạy 1 vòng
            # lặp nền retry với backoff tăng dần bất kể lần đầu thành công
            # hay không -> ở đây chỉ cần log cảnh báo rồi VẪN đi tiếp vào
            # vòng lặp chính. self.camera.read() sẽ trả về (False, None)
            # cho tới khi CameraThread tự kết nối được, ô "no signal" sẽ tự
            # biến mất trên lưới hiển thị khi đó.
            logger.warning(
                f"[{self.cam_name}] Không kết nối được lúc khởi động - "
                f"sẽ tự động thử lại ở nền, không cần khởi động lại chương trình."
            )

        logger.info(f"[{self.cam_name}] Pipeline running")

        try:
            while not self.stop_event.is_set():
                ret, frame = self.camera.read(timeout=1.0)

                if not ret:
                    if not self.camera.is_connected():
                        time.sleep(0.5)
                    continue

                self.frame_index += 1

                # -----------------------------------
                # MOTION DETECTION (gate trước YOLO) + motion hold
                # -----------------------------------
                if self.motion_detector is not None:
                    raw_motion, _ = self.motion_detector.detect(frame)

                    if raw_motion:
                        self._last_motion_frame = self.frame_index

                    # Vẫn coi là "có chuyển động" thêm MOTION_HOLD_FRAMES sau khi
                    # chuyển động thật kết thúc -> chống chập chờn, giữ ByteTrack
                    # được gọi liên tục, tránh mất track/sinh ID mới liên tục
                    motion = (self.frame_index - self._last_motion_frame) <= MOTION_HOLD_FRAMES
                else:
                    motion = True

                # -----------------------------------
                # PERSON DETECTION (YOLO) - chạy ĐỀU ĐẶN mỗi frame khi đang
                # trong vùng "có chuyển động" (không bỏ cách frame, để ByteTrack
                # không bị mất track giữa chừng)
                # -----------------------------------
                if motion:
                    with ML_INFERENCE_LOCK:
                        detections = self.detector.detect(frame)
                else:
                    detections = []

                # Vẫn cập nhật tracker mỗi frame (kể cả khi không có chuyển động)
                # để việc đếm EXIT_TIMEOUT không bị sai lệch theo thời gian thực
                current_ids = self.tracker.update(detections, frame, self.frame_index)

                if motion:
                    draw_detections(frame, detections)
                else:
                    cv2.putText(
                        frame,
                        "No motion (YOLO skipped)",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"{self.cam_name} | Tracking: {len(current_ids)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                self.output_frames[self.cam_name] = frame
        finally:
            self.tracker.finalize()
            self.camera.stop()
            logger.info(f"[{self.cam_name}] Pipeline stopped")


class ZoneCameraPipeline(threading.Thread):
    """
    Camera "thấy cả 2 chiều" (occupancy) - CHỈ đếm số người ra/vào ẨN DANH
    bằng 2 vùng polygon (ZoneCounter), KHÔNG chạy face recognition, KHÔNG
    đẩy gì vào recognition_queue - khác hẳn CameraPipeline (dùng cho camera
    cố định 1 chiều, cần định danh qua insightface).

    Vẫn dùng chung PersonDetector (YOLO + ByteTrack, đã có track_id ổn
    định) + ML_INFERENCE_LOCK để tránh segfault khi nhiều model chạy đồng
    thời (xem giải thích ML_INFERENCE_LOCK ở đầu file) - CHỈ khác ở phần xử
    lý SAU khi có detections: thay vì PersonTracker (chọn crop mặt đẹp nhất
    để gửi nhận diện), ở đây dùng ZoneCounter để phát hiện lượt băng qua
    ranh giới 2 vùng, rồi cap ẢNH TOÀN THÂN ngay tại khung hình đó, lưu
    thẳng vào zone_events (task_db) - không cần chờ track exit/timeout như
    CameraPipeline, vì thời điểm "băng qua ranh giới" đã là thời điểm rõ
    ràng để chụp, không cần chọn ảnh đẹp nhất trong nhiều frame.
    """

    def __init__(self, name, camera_url, zone_in, zone_out, output_frames, stop_event, address=None):
        super().__init__(daemon=True, name=f"ZonePipeline-{name}")

        self.cam_name = name
        self.address = address
        self.camera = CameraThread(camera_url)
        self.detector = PersonDetector(
            model_name=MODEL_PATH,
            confidence=CONFIDENCE,
            imgsz=YOLO_IMGSZ,
        )

        if ENABLE_MOTION_GATE:
            self.motion_detector = MotionDetector(
                min_area=MOTION_MIN_AREA,
                cooldown=0,
                history=MOTION_HISTORY,
                var_threshold=MOTION_VAR_THRESHOLD,
            )
        else:
            self.motion_detector = None

        self.zone_counter = ZoneCounter(zone_in, zone_out)

        self.output_frames = output_frames
        self.stop_event = stop_event
        self.frame_index = 0
        self._last_motion_frame = -999

        # Đếm CỤC BỘ (RAM, chỉ để vẽ HUD) - "Entered"/"Exited" hiển thị vẫn
        # tính từ 0 mỗi lần khởi động (chỉ đếm lượt phát sinh TRONG phiên
        # chạy này), nhưng "Current" (entered - exited) cần phản ánh ĐÚNG số
        # người đang thực sự có mặt trong cơ sở NGAY khi vừa khởi động lại,
        # không thể để về 0 rồi coi như cơ sở trống trơn trong khi thực tế
        # vẫn còn người ở trong (chương trình có thể restart bất cứ lúc nào
        # do lỗi/deploy, không có nghĩa là mọi người đã ra hết).
        #
        # Cách làm: nạp số đang có mặt THẬT (task_db.get_zone_occupancy_count,
        # đã được $inc nguyên tử vào Mongo mỗi lượt vào/ra, xem save_zone_event())
        # ngay lúc khởi động, lưu riêng vào self._initial_count (baseline) -
        # không cộng thẳng vào _entered_total để không làm sai nghĩa "Entered"
        # (chỉ số lượt vào TRONG phiên chạy này). "Current" hiển thị trên HUD
        # = self._initial_count + entered - exited (xem current_override khi
        # gọi self.zone_counter.draw() trong run()), nên đúng ngay từ frame
        # đầu tiên thay vì phải chờ có người băng qua mới đúng.
        self._initial_count = 0
        if self.address:
            try:
                self._initial_count = task_db.get_zone_occupancy_count(self.address)
            except Exception as e:
                logger.error(
                    f"[{self.cam_name}] Không lấy được số người hiện tại từ DB "
                    f"(address={self.address}), tạm coi như 0: {e}"
                )
        else:
            logger.warning(
                f"[{self.cam_name}] Camera occupancy thiếu 'address' - không thể "
                f"nạp số người hiện tại từ DB lúc khởi động, HUD sẽ bắt đầu từ 0."
            )

        logger.info(
            f"[{self.cam_name}] Số người hiện có mặt lúc khởi động "
            f"(address={self.address}): {self._initial_count}"
        )

        # entered/exited CHỈ đếm lượt phát sinh TRONG phiên chạy này (giống
        # hành vi cũ, để log HUD "Entered/Exited" không lẫn baseline vào) -
        # "Current" thật sự = self._initial_count + entered - exited, xem
        # cách dùng current_override khi gọi self.zone_counter.draw() bên dưới.
        self._entered_total = 0
        self._exited_total = 0

    def _handle_crossing_events(self, events, frame):
        h, w = frame.shape[:2]

        for ev in events:
            direction = ev["direction"]
            x1, y1, x2, y2 = ev["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]

            if direction == "in":
                self._entered_total += 1
            else:
                self._exited_total += 1

            if not ENABLE_DB_LOGGING or crop.size == 0:
                continue

            try:
                image_b64 = task_db.encode_image_to_base64(crop)
                task_db.save_zone_event(
                    self.cam_name, direction, image_b64, track_id=ev["track_id"]
                )
                logger.info(
                    f"[{self.cam_name}] [ZONE-{direction.upper()}] track {ev['track_id']}"
                )
            except Exception as e:
                logger.error(f"[{self.cam_name}] Lỗi ghi zone_event: {e}")

    def run(self):
        if not self.camera.start():
            logger.warning(
                f"[{self.cam_name}] (occupancy) Không kết nối được lúc khởi động - "
                f"sẽ tự động thử lại ở nền, không cần khởi động lại chương trình."
            )

        logger.info(f"[{self.cam_name}] Occupancy pipeline running")

        try:
            while not self.stop_event.is_set():
                ret, frame = self.camera.read(timeout=1.0)

                if not ret:
                    if not self.camera.is_connected():
                        time.sleep(0.5)
                    continue

                self.frame_index += 1

                if self.motion_detector is not None:
                    raw_motion, _ = self.motion_detector.detect(frame)

                    if raw_motion:
                        self._last_motion_frame = self.frame_index

                    motion = (self.frame_index - self._last_motion_frame) <= MOTION_HOLD_FRAMES
                else:
                    motion = True

                if motion:
                    with ML_INFERENCE_LOCK:
                        detections = self.detector.detect(frame)
                else:
                    detections = []

                events = self.zone_counter.update(detections, self.frame_index)
                if events:
                    self._handle_crossing_events(events, frame)

                if motion:
                    draw_detections(frame, detections)
                else:
                    cv2.putText(
                        frame, "No motion (YOLO skipped)", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2,
                    )

                current_in_facility = max(
                    0, self._initial_count + self._entered_total - self._exited_total
                )
                self.zone_counter.draw(
                    frame, self._entered_total, self._exited_total,
                    current_override=current_in_facility,
                )

                cv2.putText(
                    frame, f"{self.cam_name} (occupancy)", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                )

                self.output_frames[self.cam_name] = frame
        finally:
            self.camera.stop()
            logger.info(f"[{self.cam_name}] Occupancy pipeline stopped")


class RecognitionWorker(threading.Thread):
    """
    Thread nền DUY NHẤT xử lý nhận diện khuôn mặt (insightface, nặng CPU)
    và ghi log vào MongoDB. Tất cả camera-thread chỉ đẩy việc vào
    `task_queue` (cực nhanh, không chặn), thread này xử lý tuần tự phía sau
    - nhờ vậy nhận diện chạy chậm cỡ nào cũng KHÔNG làm đứng hình video camera.

    ĐÃ BỎ toàn bộ dedup theo thời gian (cooldown) và đối chiếu theo địa
    điểm (camera đối diện): hễ track nào xử lý xong và có mặt là ghi DB
    ngay (xem _finalize_event()). is_in của 1 camera được tra cứu qua
    task_db.get_camera_by_channel() và CACHE lại theo channel (không query
    Mongo mỗi lần nhận diện) - is_in chỉ còn dùng để LƯU vào face_events
    như 1 thuộc tính mô tả (vào/ra), không dùng để quyết định gì nữa.
    """

    def __init__(
        self,
        task_queue,
        face_identifier,
        stop_event,
        unknown_gallery=None,
    ):
        super().__init__(daemon=True, name="RecognitionWorker")
        self.task_queue = task_queue
        self.face_identifier = face_identifier
        self.stop_event = stop_event
        # UnknownGallery: cấp/nhận lại temp_id ổn định cho người LẠ (chưa
        # enroll) - xem UNKNOWN_GALLERY_* phía trên. Chỉ dùng để BIẾT đó là
        # ai (ghi kèm vào face_events), KHÔNG dùng để chặn/gộp việc ghi DB
        # nữa. None = tắt tính năng (mọi người lạ đều "vô danh").
        self.unknown_gallery = unknown_gallery

        # cache channel -> is_in, tránh query Mongo mỗi lần nhận diện
        self._channel_is_in_cache = {}

    def _get_is_in(self, cam_name):
        if cam_name in self._channel_is_in_cache:
            return self._channel_is_in_cache[cam_name]

        cam_doc = task_db.get_camera_by_channel(cam_name)
        is_in = cam_doc.get("is_in") if cam_doc else None

        if is_in is None:
            logger.warning(
                f"[{cam_name}] Không xác định được is_in của camera này "
                f"(thiếu trong DB?) - chỉ ảnh hưởng giá trị mô tả lưu vào "
                f"face_events, không ảnh hưởng việc có ghi DB hay không."
            )

        self._channel_is_in_cache[cam_name] = is_in
        return is_in

    def run(self):
        while not self.stop_event.is_set():
            try:
                cam_name, track_id, crops, disappear_time = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self.face_identifier is None:
                logger.info(f"[{cam_name}] [SAVED] Track {track_id}")
                self._finalize_event(None, cam_name, self._get_is_in(cam_name), None, None, crops)
                continue

            with ML_INFERENCE_LOCK:
                person, score, face_crop, embedding = self.face_identifier.identify_from_candidates(crops)
            is_in = self._get_is_in(cam_name)

            if person:
                # Người ĐÃ enroll - person_id thật từ MongoDB persons, luôn
                # ổn định qua mọi lần xuất hiện (không cần UnknownGallery).
                identity = person["person_id"]
                logger.info(f"[{cam_name}] [MATCH] Track {track_id} -> "
                            f"{person['name']} (person_id={identity}, is_in={is_in}, {score:.3f})")
            elif self.unknown_gallery is not None and embedding is not None:
                # Người LẠ (chưa enroll) - tra/cấp temp_id ổn định qua
                # UnknownGallery để "nhớ" họ nếu quay lại sau đó (chỉ dùng
                # để nhận DIỆN họ là ai, KHÔNG dùng để chặn ghi DB).
                identity = self.unknown_gallery.identify_or_register(embedding)
                logger.info(f"[{cam_name}] [UNKNOWN] Track {track_id} -> {identity} "
                            f"(is_in={is_in}, {score:.3f})")
            else:
                identity = None
                logger.info(f"[{cam_name}] [UNKNOWN] Track {track_id} ({score:.3f})")

            # KHÔNG còn treo lại chờ đối chiếu camera đối diện, KHÔNG còn
            # chặn theo cooldown thời gian - hễ track này xử lý xong là ghi
            # thẳng vào DB ngay (xem _finalize_event()).
            self._finalize_event(identity, cam_name, is_in, person, face_crop, crops)

    def _finalize_event(self, identity, cam_name, is_in, person, face_crop, crops):
        """Điểm ghi DB THẬT SỰ duy nhất. HỄ CÓ MẶT LÀ GHI NGAY - không còn
        kiểm tra thời gian (cooldown) hay đối chiếu địa điểm (camera đối
        diện) trước khi ghi nữa. identity (person_id thật hoặc temp_id từ
        UnknownGallery) chỉ dùng để LƯU vào face_events cho biết đó là ai,
        không còn dùng để chặn/gộp sự kiện.

        QUAN TRỌNG: nếu insightface không tìm ra mặt ở BẤT KỲ crop ứng viên
        nào của track này (face_crop=None), vẫn BỎ QUA - không lưu ảnh toàn
        thân, vì không dùng được để xác minh lại bằng mắt hay để enroll.
        """
        if face_crop is None:
            logger.warning(
                f"[{cam_name}] Không detect được mặt ở bất kỳ crop ứng viên "
                f"nào của track này -> BỎ QUA, không ghi DB"
            )
            return

        if ENABLE_DB_LOGGING:
            self._log_to_db(cam_name, person, face_crop, identity)
            self._update_session(cam_name, is_in, identity, person)

    def _update_session(self, cam_name, is_in, identity, person):
        """
        Cập nhật person_sessions (trạng thái đang ở trong/ngoài) - TÁCH
        BIỆT với việc ghi face_events ở trên (đó là log thô mọi lần thấy
        mặt, cái này là trạng thái "phiên" vào/ra để đếm số lượng).

        - identity=None (không xác định được là ai) -> không có khoá ổn
          định để mở/đóng session -> bỏ qua, không ảnh hưởng face_events.
        - is_in=None (camera thiếu cấu hình is_in trong DB) -> không biết
          đây là chiều vào hay ra -> bỏ qua, chỉ log cảnh báo 1 lần (đã log
          ở _get_is_in()).
        - is_in=True  (camera IN)  -> handle_entry(): CHỈ tạo session mới
          khi CHƯA có session đang mở, có rồi thì bỏ qua (chống đếm trùng
          khi đứng trước camera IN nhiều frame liên tiếp).
        - is_in=False (camera OUT) -> handle_exit(): LUÔN cập nhật, không
          bao giờ bỏ qua hoàn toàn - đóng session đang mở (lần OUT đầu
          tiên sau khi vào), hoặc nếu đã "outside"/chưa từng có session
          thì vẫn cập nhật mốc thời gian thấy gần nhất. Chỉ khi gặp lại
          camera IN mới tạo session MỚI - đúng yêu cầu "liên tục update
          cam_out cho đến khi gặp cam in thì tạo session mới".
        """
        if identity is None or is_in is None:
            return

        name = person["name"] if person else None
        person_type = person.get("type") if person else "nguoi_la"

        try:
            if is_in:
                session = task_db.handle_entry(
                    identity, channel=cam_name, name=name, person_type=person_type
                )
                if session is not None:
                    logger.info(
                        f"[{cam_name}] [SESSION-IN] {name or identity} -> "
                        f"entry_time={session['entry_time'].strftime('%H:%M')}"
                    )
                else:
                    logger.debug(
                        f"[{cam_name}] [SESSION-IN] {name or identity} vẫn đang "
                        f"'inside' -> bỏ qua, không tạo session trùng"
                    )
            else:
                session = task_db.handle_exit(
                    identity, channel=cam_name, name=name, person_type=person_type
                )
                logger.info(
                    f"[{cam_name}] [SESSION-OUT] {name or identity} -> "
                    f"exit_time={session['exit_time'].strftime('%H:%M')}"
                )
        except Exception as e:
            logger.error(f"[{cam_name}] [SESSION ERROR] Không cập nhật được person_sessions: {e}")

    def _log_to_db(self, cam_name, person, crop, identity):
        if crop is None:
            return

        try:
            image_b64 = task_db.encode_image_to_base64(crop)

            if person:
                face_recognition = {
                    "result": "matched",
                    "person_id": person["person_id"],
                    "name": person["name"],
                    "type": person.get("type") or "known",
                }
            else:
                face_recognition = {"result": "unmatched"}

            face_payload = {
                "face_recognition": face_recognition,
                "StartTime": time.time(),
                "Image2": image_b64,
                "channel": cam_name,
                # identity = person_id thật (đã enroll) hoặc temp_id ổn định
                # từ UnknownGallery (người lạ) hoặc None (không detect được
                # mặt) - xem giải thích chi tiết ở task_db.save_face_to_db().
                # BẮT BUỘC truyền xuống để get_current_staying() đếm đúng
                # từng người lạ riêng biệt thay vì gộp chung person_id=-1.
                "identity": identity,
            }

            inserted_id = task_db.save_face_to_db(face_payload)
            
            if inserted_id is None:
                # save_face_to_db đã tự buffer ra đĩa khi Mongo lỗi (xem
                # task_db.py) - không mất event, nhưng vẫn nên log rõ ở đây.
                logger.warning("[%s] Event bị buffer tạm ra đĩa do Mongo lỗi", cam_name)
            else:
                name= face_recognition.get("name") if face_recognition.get("name") else "Người lạ"
                logger.info(f"Save to database success {name}")
        except Exception as e:
            logger.error("[%s] [DB ERROR] Không lưu được event: %s", cam_name, e)


def main():
    # Set 1 LẦN DUY NHẤT cho toàn tiến trình - không qset lại trong từng camera
    torch.set_num_threads(TORCH_NUM_THREADS)

    identity_cams_cfg = load_identity_cameras_from_db()
    occupancy_cams_cfg = load_occupancy_cameras_from_db()

    if not identity_cams_cfg and not occupancy_cams_cfg:
        logger.error("Không có camera nào (status=True) trong DB.")
        logger.error("Thêm bằng: python manage_cameras.py add --serial ... "
              "--channel ... --url ... --in/--out")
        logger.error("Camera occupancy (2 chiều) cấu hình zone bằng: "
              "python configure_zone_camera.py --channel ...")
        return

    stop_event = threading.Event()
    output_frames = {}  # {cam_name: latest_frame} - chỉ đọc/ghi ở đây, an toàn với GIL

    face_identifier = None
    unknown_gallery = None

    # Chỉ tải model insightface (nặng) nếu THẬT SỰ có ít nhất 1 camera
    # identity - camera occupancy không bao giờ dùng tới face_identifier,
    # tải lên vô ích nếu 1 site chỉ toàn camera occupancy.
    if ENABLE_FACE_ID and identity_cams_cfg:
        logger.info("Loading face identifier (insightface + faiss, dùng chung cho tất cả camera)...")
        face_identifier = PersonDBRecognizer(
            ctx_id=FACE_CTX_ID,
            threshold=FACE_THRESHOLD,
            refresh_interval=FACE_REFRESH_INTERVAL,
        )
        logger.info("Face identifier ready.")

        # Bộ nhớ tạm cho người LẠ (chưa enroll) - xem UNKNOWN_GALLERY_* phía trên.
        unknown_gallery = UnknownGallery(
            threshold=UNKNOWN_GALLERY_THRESHOLD,
            ttl_seconds=UNKNOWN_GALLERY_TTL_SECONDS,
        )
        logger.info(f"Unknown-person gallery ready (ttl={UNKNOWN_GALLERY_TTL_SECONDS}s).")

    # Hàng đợi việc nhận diện - mọi camera-thread IDENTITY đẩy vào đây,
    # KHÔNG tự gọi insightface, để tránh làm đứng hình lúc có người EXIT.
    # Camera occupancy KHÔNG dùng hàng đợi này (xử lý + ghi DB thẳng trong
    # ZoneCameraPipeline, vì không cần insightface).
    recognition_queue = queue.Queue()
    recognition_worker = None
    if identity_cams_cfg:
        recognition_worker = RecognitionWorker(
            recognition_queue, face_identifier, stop_event,
            unknown_gallery=unknown_gallery,
        )
        recognition_worker.start()

    workers = []

    for cam in identity_cams_cfg:
        worker = CameraPipeline(
            name=cam["name"],
            camera_url=cam["url"],
            output_frames=output_frames,
            stop_event=stop_event,
            recognition_queue=recognition_queue,
        )
        worker.start()
        workers.append(worker)

    for cam in occupancy_cams_cfg:
        worker = ZoneCameraPipeline(
            name=cam["name"],
            camera_url=cam["url"],
            zone_in=cam["zone_in"],
            zone_out=cam["zone_out"],
            output_frames=output_frames,
            stop_event=stop_event,
            address=cam.get("address"),
        )
        worker.start()
        workers.append(worker)

    print("===================================")
    logger.info(
        f"{len(identity_cams_cfg)} identity + {len(occupancy_cams_cfg)} occupancy "
        f"camera pipeline(s) started - Press Q to quit"
    )
    print("===================================")

    cam_names = [cam["name"] for cam in identity_cams_cfg] + \
                [cam["name"] for cam in occupancy_cams_cfg]

    # Tạo cửa sổ trước, cho phép resize/fullscreen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    is_fullscreen = FULLSCREEN_ON_START
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
    )

    last_flush = time.time()

    try:


        while True:
            grid = build_grid(output_frames, cam_names)

            if grid is not None:
                cv2.imshow(WINDOW_NAME, grid)

            # Ghi bù định kỳ các face_event từng bị lỗi ghi Mongo (buffer
            # ra đĩa bởi task_db.save_face_to_db khi mất kết nối tạm thời)
            if ENABLE_DB_LOGGING and (time.time() - last_flush) >= PENDING_FLUSH_INTERVAL:
                try:
                    task_db.flush_pending_face_events()
                except Exception as e:
                    logger.error("Lỗi khi flush pending face_events: %s", e)
                last_flush = time.time()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                stop_event.set()
                break
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                )
    finally:

        for w in workers:
            w.join(timeout=5)
        if recognition_worker is not None:
            recognition_worker.join(timeout=5)
        if face_identifier is not None:
            face_identifier.stop()
        if unknown_gallery is not None:
            unknown_gallery.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()