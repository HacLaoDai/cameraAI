import threading

import torch
from ultralytics import YOLO

# -------------------------------------------------------------------------
# LOCK TOÀN CỤC (dùng chung cho MỌI instance PersonDetector trong tiến
# trình) - QUAN TRỌNG khi chạy NHIỀU camera (nhiều PersonDetector, mỗi cái
# 1 model YOLO riêng) trong NHIỀU THREAD khác nhau cùng lúc.
#
# Chạy nhiều model PyTorch/YOLO trên CPU THẬT SỰ ĐỒNG THỜI ở nhiều thread
# khác nhau trong CÙNG 1 tiến trình là nguyên nhân phổ biến gây CRASH CẤP
# THẤP (segmentation fault) - không có traceback Python nào cả, chương
# trình chỉ đột ngột dừng - vì các thư viện BLAS bên dưới (OpenBLAS/MKL)
# mà torch dùng để tính toán không đảm bảo an toàn khi bị gọi chồng chéo
# từ nhiều thread thật sự song song.
#
# Lock này đảm bảo tại 1 thời điểm CHỈ CÓ ĐÚNG 1 lệnh .track() (của bất kỳ
# camera nào) được chạy - các camera khác đang chờ tới lượt sẽ CHỜ Ở ĐÂY
# (rất nhanh, vài chục-vài trăm ms mỗi lần), KHÔNG bị mất frame, chỉ hơi
# giảm FPS hiệu dụng khi nhiều camera cùng có chuyển động một lúc - đánh
# đổi hợp lý để đổi lấy việc không crash.
_YOLO_INFERENCE_LOCK = threading.Lock()


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

        # QUAN TRỌNG: bọc lock quanh ĐÚNG lệnh gọi vào model - xem giải
        # thích ở _YOLO_INFERENCE_LOCK phía trên. Đây là điểm duy nhất
        # PyTorch/YOLO thực sự chạy tính toán nặng, các bước xử lý
        # kết quả (result.boxes...) bên dưới không cần khoá.
        with _YOLO_INFERENCE_LOCK:
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