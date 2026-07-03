from ultralytics import YOLO


class PersonDetector:

    def __init__(
        self,
        model_name="yolov8n.pt",
        confidence=0.5,
    ):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                if box.id is None:
                    continue

                detections.append(
                    {
                        "track_id": int(box.id.item()),
                        "bbox": list(
                            map(
                                int,
                                box.xyxy[0]
                            )
                        ),
                        "confidence": float(
                            box.conf.item()
                        ),
                    }
                )

        return detections