import os
import math
import time
import threading

import cv2
import numpy as np

from camera.CameraThread import CameraThread
from detectors.PersonDetector import PersonDetector
from detectors.motion_detector import MotionDetector
from trackers.person_tracker import PersonTracker
from recognition.face_identifier import FaceIdentifier

# ===============================
# CONFIG
# ===============================

CAMERAS = [
    
    {"name": "webcam", "url": 0}, 
    # {"name": "cam1", "url": "rtsp://localhost:8554/live"},
    # {"name": "cam2", "url": "rtsp://localhost:8554/live4"},
    # Thêm bao nhiêu camera tùy ý ở đây
]

MODEL_PATH = os.path.join("detectors", "yolov8n.pt")
CONFIDENCE = 0.5

SAVE_DIR = "saved_person"
EXIT_TIMEOUT = 30

# Motion Detection - lọc trước khi chạy YOLO (đúng theo sơ đồ kiến trúc)
ENABLE_MOTION_GATE = True
MOTION_MIN_AREA = 5000
MOTION_HISTORY = 500
MOTION_VAR_THRESHOLD = 16

ENABLE_FACE_ID = True
FACE_DB_FOLDER = "/home/lychien/Desktop/img"
FACE_INDEX_FILE = "faces.index"
FACE_PATH_FILE = "paths.pkl"
FACE_THRESHOLD = 0.62

# Hiển thị dạng lưới - kích thước mỗi ô (mỗi camera) trong lưới
GRID_CELL_WIDTH = 480
GRID_CELL_HEIGHT = 270


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
        face_identifier=None,
        face_lock=None,
    ):
        super().__init__(daemon=True, name=f"Pipeline-{name}")

        self.cam_name = name
        self.camera = CameraThread(camera_url)
        self.detector = PersonDetector(model_name=MODEL_PATH, confidence=CONFIDENCE)

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

        self.face_identifier = face_identifier
        self.face_lock = face_lock

        self.output_frames = output_frames
        self.stop_event = stop_event
        self.frame_index = 0

        self.tracker = PersonTracker(
            save_dir=os.path.join(SAVE_DIR, name),
            exit_timeout=EXIT_TIMEOUT,
            on_person_saved=self._on_person_saved,
        )

    def _on_person_saved(self, track_id, crop):
        if self.face_identifier is None:
            print(f"[{self.cam_name}] [SAVED] Track {track_id}")
            return

        # Khóa lại vì FaceIdentifier được nhiều camera-thread dùng chung
        with self.face_lock:
            name, score = self.face_identifier.identify(crop)

        if name:
            print(f"[{self.cam_name}] [MATCH] Track {track_id} -> {name} ({score:.3f})")
        else:
            print(f"[{self.cam_name}] [UNKNOWN] Track {track_id} ({score:.3f})")

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
                # MOTION DETECTION (gate trước YOLO)
                # -----------------------------------
                if self.motion_detector is not None:
                    motion, _ = self.motion_detector.detect(frame)
                else:
                    motion = True

                # -----------------------------------
                # PERSON DETECTION (YOLO) - chỉ chạy khi có chuyển động
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


def main():
    stop_event = threading.Event()
    output_frames = {}  # {cam_name: latest_frame} - chỉ đọc/ghi ở đây, an toàn với GIL

    face_identifier = None
    face_lock = threading.Lock()

    if ENABLE_FACE_ID:
        print("[INFO] Loading face identifier (dùng chung cho tất cả camera)...")
        face_identifier = FaceIdentifier(
            db_folder=FACE_DB_FOLDER,
            index_file=FACE_INDEX_FILE,
            path_file=FACE_PATH_FILE,
            threshold=FACE_THRESHOLD,
        )
        print("[INFO] Face identifier ready.")

    workers = []
    for cam in CAMERAS:
        worker = CameraPipeline(
            name=cam["name"],
            camera_url=cam["url"],
            output_frames=output_frames,
            stop_event=stop_event,
            face_identifier=face_identifier,
            face_lock=face_lock,
        )
        worker.start()
        workers.append(worker)

    print("===================================")
    print(f"{len(workers)} camera pipeline(s) started - Press Q to quit")
    print("===================================")

    cam_names = [cam["name"] for cam in CAMERAS]

    try:
        while True:
            grid = build_grid(output_frames, cam_names)

            if grid is not None:
                cv2.imshow("Multi-Camera Grid", grid)

            key = cv2.waitKey(1)
            if key == ord("q"):
                stop_event.set()
                break
    finally:
        for w in workers:
            w.join(timeout=5)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()