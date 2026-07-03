import cv2

from detectors.PersonDetector import (
    PersonDetector,
)

from trackers.person_tracker import (
    PersonTracker,
)

RTSP = (
    "rtsp://localhost:8554/live4"
)

cap = cv2.VideoCapture(
    RTSP
)

detector = PersonDetector()

tracker = PersonTracker()

frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_index += 1

    detections = detector.detect(
        frame
    )

    tracker.update(
        detections,
        frame,
        frame_index,
    )

    for det in detections:

        x1, y1, x2, y2 = (
            det["bbox"]
        )

        track_id = det[
            "track_id"
        ]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            str(track_id),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.imshow(
        "Tracking",
        frame,
    )

    if (
        cv2.waitKey(1)
        == ord("q")
    ):
        break

tracker.finalize()