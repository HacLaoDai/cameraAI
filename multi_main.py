import os
import math
import time
import threading
import queue

import cv2
import numpy as np
import torch

from camera.CameraThread import CameraThread
from detectors.PersonDetector import PersonDetector
from detectors.motion_detector import MotionDetector
from trackers.person_tracker import PersonTracker
from recognition.person_db_recognizer import PersonDBRecognizer
from database import task_db

# ===============================
# CONFIG
# ===============================

# Danh sách camera KHÔNG còn khai báo cứng ở đây nữa - được load động từ
# MongoDB (collection `cameras`, chỉ lấy camera có status=True) mỗi khi
# multi_main.py khởi động, xem hàm load_cameras_from_db() bên dưới.
#
# Thêm / sửa / xóa / tạm tắt camera bằng:
#     python manage_cameras.py add    --serial ... --channel ... --url ... --in/--out
#     python manage_cameras.py edit   --channel ... --url ...
#     python manage_cameras.py delete --channel ...
#     python manage_cameras.py list
#
# Sau khi thêm/sửa/xóa cần KHỞI ĐỘNG LẠI multi_main.py để nhận cấu hình mới.

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

# Motion Detection - lọc trước khi chạy YOLO (đúng theo sơ đồ kiến trúc)
ENABLE_MOTION_GATE = True
MOTION_MIN_AREA = 5000
MOTION_HISTORY = 500
MOTION_VAR_THRESHOLD = 16

# Nhận diện khuôn mặt: insightface (ArcFace) + faiss, đọc/refresh trực tiếp
# từ MongoDB persons - đúng bằng embedding mà manage_persons.py enroll vào DB
# (xem recognition/person_db_recognizer.py).
ENABLE_FACE_ID = True
FACE_CTX_ID = -1          # -1 = CPU, 0 = GPU đầu tiên
FACE_THRESHOLD = 0.45     # ngưỡng cosine similarity - cần tự test/tinh chỉnh
FACE_REFRESH_INTERVAL = 60  # giây - tự load lại persons từ DB sau mỗi khoảng này

# Ghi log ra/vào vào MongoDB (task_db.py)
# Yêu cầu: channel của camera (khai báo bằng manage_cameras.py) TRÙNG với
# channel trong DB, và đúng is_in=True (cổng vào) / False (cổng ra)
ENABLE_DB_LOGGING = True

# Hiển thị dạng lưới - kích thước mỗi ô (mỗi camera) trong lưới
GRID_CELL_WIDTH = 480
GRID_CELL_HEIGHT = 270

WINDOW_NAME = "Multi-Camera Grid"
FULLSCREEN_ON_START = True


def load_cameras_from_db():
    """
    Đọc danh sách camera đang bật (status=True) từ MongoDB, trả về đúng
    format mà main() cần: [{"name": <channel>, "url": <rtsp_url hoặc int>}].

    "name" lấy từ field "channel" trong DB - PHẢI khớp với channel dùng khi
    ghi face_events (task_db.save_face_to_db), để tính năng vào/ra hoạt động
    đúng (xem task_db.get_current_staying()/get_person_movements()).
    """
    docs = task_db.get_active_cameras()

    cams = []

    for d in docs:
        url = d.get("url")
        channel = d.get("channel")

        if not channel:
            print(f"[WARN] Bỏ qua camera thiếu channel (_id={d.get('_id')})")
            continue

        if url is None or url == "":
            print(f"[WARN] Camera '{channel}' chưa có url trong DB, bỏ qua "
                  f"(sửa bằng: python manage_cameras.py edit --channel {channel} --url ...)")
            continue

        # url lưu trong Mongo luôn là string - nếu là số (vd "0" cho webcam
        # local) thì convert sang int cho cv2.VideoCapture, còn RTSP thì giữ
        # nguyên string.
        if isinstance(url, str) and url.strip().lstrip("-").isdigit():
            url = int(url)

        cams.append({"name": channel, "url": url})

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
            cell = cv2.resize(frame, (cell_width, cell_height))

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
            exit_timeout=EXIT_TIMEOUT,
            on_person_saved=self._on_person_saved,
        )

    def _on_person_saved(self, track_id, crops):
        # CHỈ đẩy vào hàng đợi - không gọi insightface ở đây, để không chặn
        # (block) luồng đọc/xử lý camera này. Việc nhận diện + ghi DB thật sự
        # được xử lý ở RecognitionWorker (thread nền riêng, xem bên dưới).
        self.recognition_queue.put((self.cam_name, track_id, crops))

    def run(self):
        if not self.camera.start():
            print(f"[{self.cam_name}] Cannot start camera")
            return

        print(f"[{self.cam_name}] Pipeline running")

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
            print(f"[{self.cam_name}] Pipeline stopped")


class RecognitionWorker(threading.Thread):
    """
    Thread nền DUY NHẤT xử lý nhận diện khuôn mặt (insightface, nặng CPU)
    và ghi log vào MongoDB. Tất cả camera-thread chỉ đẩy việc vào
    `task_queue` (cực nhanh, không chặn), thread này xử lý tuần tự phía sau
    - nhờ vậy nhận diện chạy chậm cỡ nào cũng KHÔNG làm đứng hình video camera.
    """

    def __init__(self, task_queue, face_identifier, stop_event):
        super().__init__(daemon=True, name="RecognitionWorker")
        self.task_queue = task_queue
        self.face_identifier = face_identifier
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                cam_name, track_id, crops = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            person = None
            score = 0.0

            if self.face_identifier is not None:
                person, score = self.face_identifier.identify_from_candidates(crops)

                if person:
                    print(f"[{cam_name}] [MATCH] Track {track_id} -> "
                          f"{person['name']} (person_id={person['person_id']}, {score:.3f})")
                else:
                    print(f"[{cam_name}] [UNKNOWN] Track {track_id} ({score:.3f})")
            else:
                print(f"[{cam_name}] [SAVED] Track {track_id}")

            if ENABLE_DB_LOGGING:
                best_crop = crops[0] if crops else None
                self._log_to_db(cam_name, person, best_crop)

    def _log_to_db(self, cam_name, person, crop):
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
            }

            task_db.save_face_to_db(face_payload)
        except Exception as e:
            print(f"[{cam_name}] [DB ERROR] Không lưu được event: {e}")


def main():
    # Set 1 LẦN DUY NHẤT cho toàn tiến trình - không set lại trong từng camera
    torch.set_num_threads(TORCH_NUM_THREADS)

    cameras_cfg = load_cameras_from_db()

    if not cameras_cfg:
        print("[ERROR] Không có camera nào (status=True) trong DB.")
        print("        Thêm bằng: python manage_cameras.py add --serial ... "
              "--channel ... --url ... --in/--out")
        return

    stop_event = threading.Event()
    output_frames = {}  # {cam_name: latest_frame} - chỉ đọc/ghi ở đây, an toàn với GIL

    face_identifier = None

    if ENABLE_FACE_ID:
        print("[INFO] Loading face identifier (insightface + faiss, dùng chung cho tất cả camera)...")
        face_identifier = PersonDBRecognizer(
            ctx_id=FACE_CTX_ID,
            threshold=FACE_THRESHOLD,
            refresh_interval=FACE_REFRESH_INTERVAL,
        )
        print("[INFO] Face identifier ready.")

    # Hàng đợi việc nhận diện - mọi camera-thread đẩy vào đây, KHÔNG tự gọi
    # insightface, để tránh làm đứng hình lúc có người EXIT
    recognition_queue = queue.Queue()
    recognition_worker = RecognitionWorker(recognition_queue, face_identifier, stop_event)
    recognition_worker.start()

    workers = []
    for cam in cameras_cfg:
        worker = CameraPipeline(
            name=cam["name"],
            camera_url=cam["url"],
            output_frames=output_frames,
            stop_event=stop_event,
            recognition_queue=recognition_queue,
        )
        worker.start()
        workers.append(worker)

    print("===================================")
    print(f"{len(workers)} camera pipeline(s) started - Press Q to quit")
    print("===================================")

    cam_names = [cam["name"] for cam in cameras_cfg]

    # Tạo cửa sổ trước, cho phép resize/fullscreen
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    is_fullscreen = FULLSCREEN_ON_START
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
    )

    try:
        while True:
            grid = build_grid(output_frames, cam_names)

            if grid is not None:
                cv2.imshow(WINDOW_NAME, grid)

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
        recognition_worker.join(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()