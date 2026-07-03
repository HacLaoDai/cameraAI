import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

# ===============================
# CONFIG
# ===============================

MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.5

# CAMERA = 0
CAMERA = "rtsp://localhost:8554/live4"

TRACKER = "bytetrack.yaml"

SAVE_DIR = "saved_person"
# How many frames without a detection before we consider the person gone
EXIT_TIMEOUT = 30


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ===============================
    # LOAD MODEL / OPEN CAMERA
    # ===============================
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tracks = {}
    frame_index = 0

    print("===================================")
    print("YOLO + ByteTrack Started")
    print("Press Q to Quit")
    print("===================================")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1

            # ---------------------------------------
            # YOLO Tracking
            # ---------------------------------------
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER,
                classes=[0],
                conf=CONFIDENCE,
                verbose=False,
            )

            current_ids = set()

            # ---------------------------------------
            # LOOP DETECTIONS
            # ---------------------------------------
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    # Sometimes ID is None (not yet assigned by tracker)
                    if box.id is None:
                        continue

                    track_id = int(box.id.item())
                    current_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf.item())
                    area = (x2 - x1) * (y2 - y1)

                    # -----------------------------------
                    # NEW PERSON -> create the entry FIRST
                    # -----------------------------------
                    if track_id not in tracks:
                        tracks[track_id] = {
                            "first_frame": frame_index,
                            "last_frame": frame_index,
                            "enter_time": datetime.now(),
                            "last_seen_time": datetime.now(),
                            "last_box": (x1, y1, x2, y2),
                            "best_area": 0,
                            "best_frame": None,
                            "best_box": None,
                            "saved": False,
                        }
                        print(f"[ENTER] Person {track_id}")

                    track = tracks[track_id]

                    # -----------------------------------
                    # UPDATE existing/new entry
                    # -----------------------------------
                    track["last_frame"] = frame_index
                    track["last_seen_time"] = datetime.now()
                    track["last_box"] = (x1, y1, x2, y2)

                    if area > track["best_area"]:
                        track["best_area"] = area
                        track["best_frame"] = frame.copy()
                        track["best_box"] = (x1, y1, x2, y2)

                    # -----------------------------------
                    # DRAW
                    # -----------------------------------
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            # ---------------------------------------
            # HANDLE EXIT (people not seen for EXIT_TIMEOUT frames)
            # ---------------------------------------
            handle_exits(tracks, current_ids, frame_index)

            # ---------------------------------------
            # HUD + DISPLAY
            # ---------------------------------------
            cv2.putText(
                frame,
                f"Tracking: {len(current_ids)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Tracking", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ---------------------------------------
        # FLUSH: treat everyone still being tracked
        # as an exit once the stream/window closes
        # ---------------------------------------
        finalize_remaining_tracks(tracks)


def handle_exits(tracks, current_ids, frame_index):
    """Detect tracks that have disappeared for too long, save their best
    crop, log the exit, and remove them from the active tracks dict."""
    lost_ids = []

    for track_id, info in tracks.items():
        if track_id in current_ids:
            continue

        if frame_index - info["last_frame"] > EXIT_TIMEOUT:
            info["exit_time"] = datetime.now()
            duration = (info["exit_time"] - info["enter_time"]).total_seconds()

            print()
            print("============================")
            print(f"[EXIT] Track {track_id}")
            print(f"Enter : {info['enter_time']}")
            print(f"Exit  : {info['exit_time']}")
            print(f"Time  : {duration:.1f} sec")

            save_best_crop(track_id, info)

            print("============================")
            lost_ids.append(track_id)

    for track_id in lost_ids:
        del tracks[track_id]


def save_best_crop(track_id, info):
    """Save the best (largest bounding-box) frame captured for this track."""
    if info["saved"] or info["best_frame"] is None:
        return

    x1, y1, x2, y2 = info["best_box"]
    img = info["best_frame"]
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    person = img[y1:y2, x1:x2]
    if person.size == 0:
        return

    filename = os.path.join(SAVE_DIR, f"person_{track_id}.jpg")
    cv2.imwrite(filename, person)
    print("Saved :", filename)
    info["saved"] = True


def finalize_remaining_tracks(tracks):
    """Called once the video loop ends: save/log anyone still active."""
    for track_id, info in list(tracks.items()):
        info.setdefault("exit_time", datetime.now())
        duration = (info["exit_time"] - info["enter_time"]).total_seconds()

        print()
        print("============================")
        print(f"[EXIT] Track {track_id} (stream ended)")
        print(f"Enter : {info['enter_time']}")
        print(f"Exit  : {info['exit_time']}")
        print(f"Time  : {duration:.1f} sec")

        save_best_crop(track_id, info)

        print("============================")


# if __name__ == "__main__":
#     main()