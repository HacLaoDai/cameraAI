import os
import time

import cv2

from camera.CameraThread import CameraThread
from detectors.PersonDetector import PersonDetector
from trackers.person_tracker import PersonTracker
from recognition.face_identifier import FaceIdentifier

# ===============================
# CONFIG
# ===============================

CAMERA_URL = "rtsp://localhost:8554/live4"
MODEL_PATH = os.path.join("detectors", "yolov8n.pt")
CONFIDENCE = 0.5

SAVE_DIR = "saved_person"
EXIT_TIMEOUT = 30  # số frame không thấy thì coi là EXIT

# Nhận diện khuôn mặt - index đã được build sẵn (recognizer.build_index())
FACE_DB_FOLDER = "/home/lychien/Desktop/img"
FACE_INDEX_FILE = "faces.index"
FACE_PATH_FILE = "paths.pkl"
FACE_THRESHOLD = 0.62


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
            0.7,
            (0, 0, 255),
            2,
        )


def main():
    camera = CameraThread(CAMERA_URL)

    if not camera.start():
        raise RuntimeError("Cannot start camera")

    detector = PersonDetector(model_name=MODEL_PATH, confidence=CONFIDENCE)

    print("[INFO] Loading face identifier (insightface + deepface index)...")
    face_identifier = FaceIdentifier(
        db_folder=FACE_DB_FOLDER,
        index_file=FACE_INDEX_FILE,
        path_file=FACE_PATH_FILE,
        threshold=FACE_THRESHOLD,
    )
    print("[INFO] Face identifier ready.")

    def handle_person_saved(track_id, crop):
        """Được PersonTracker gọi ngay khi 1 người EXIT và ảnh đẹp nhất đã lưu."""
        name, score = face_identifier.identify(crop)

        if name:
            print(f"[MATCH] Track {track_id} -> {name} (score={score:.3f})")
        else:
            print(f"[UNKNOWN] Track {track_id} (score={score:.3f})")

    tracker = PersonTracker(
        save_dir=SAVE_DIR,
        exit_timeout=EXIT_TIMEOUT,
        on_person_saved=handle_person_saved,
    )

    frame_index = 0

    print("===================================")
    print("Pipeline started - Press Q to quit")
    print("===================================")

    try:
        while True:
            ret, frame = camera.read(timeout=1.0)

            if not ret:
                if not camera.is_connected():
                    print("[WARN] Camera not connected, waiting...")
                    time.sleep(0.5)
                continue

            frame_index += 1

            # -----------------------------------
            # DETECT + TRACK
            # -----------------------------------
            detections = detector.detect(frame)
            current_ids = tracker.update(detections, frame, frame_index)

            # -----------------------------------
            # DRAW
            # -----------------------------------
            draw_detections(frame, detections)

            cv2.putText(
                frame,
                f"Tracking: {len(current_ids)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Pipeline", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    finally:
        tracker.finalize()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()