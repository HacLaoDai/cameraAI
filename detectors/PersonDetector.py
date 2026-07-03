import torch
from ultralytics import YOLO


class PersonDetector:

    def __init__(
        self,
        model_name="yolov8n.pt",
        confidence=0.5,
        imgsz=416,          # giảm từ 640 (mặc định) -> tăng tốc đáng kể trên CPU
        num_threads=None,   # giới hạn số luồng CPU cho torch (None = giữ mặc định)
    ):
        if num_threads is not None:
            # Lưu ý: đây là cấu hình TOÀN TIẾN TRÌNH (global), không phải riêng
            # cho từng camera. Nếu chạy nhiều camera, nên set 1 lần duy nhất
            # ở main() thay vì gọi lặp lại ở từng PersonDetector.
            torch.set_num_threads(num_threads)

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.imgsz = imgsz

    def detect(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            imgsz=self.imgsz,
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