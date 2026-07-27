"""
⚠️ DEPRECATED - KHÔNG còn được dùng trong pipeline chính (multi_main.py).

Xem ghi chú ở recognition/build_system_regconition.py - pipeline hiện tại
dùng recognition/person_db_recognizer.py (PersonDBRecognizer) thay cho
FaceIdentifier ở file này.
"""

import os
import tempfile

import cv2

# Import build_system_regconition TRƯỚC để biến môi trường CUDA_VISIBLE_DEVICES=-1
# được set trước khi insightface/onnxruntime khởi tạo session (tránh xung đột GPU/CPU).
from recognition.build_system_regconition import ArcFaceRecognizer
from detectors.detect_face import ArcFaceExtractor


class FaceIdentifier:
    """
    Bước 1 (best-effort): dùng insightface (ArcFaceExtractor) để TÌM vị trí
            khuôn mặt trong ảnh người, giúp cắt gọn ảnh trước khi so khớp.
    Bước 2: đưa ảnh (đã cắt gọn nếu bước 1 thành công, hoặc nguyên crop người
            nếu không) cho ArcFaceRecognizer (DeepFace + faiss) để so khớp.

    QUAN TRỌNG: insightface ở bước 1 chỉ mang tính hỗ trợ, KHÔNG phải điều
    kiện bắt buộc. Nếu insightface không tìm thấy mặt (dùng detector SCRFD,
    khá nhạy với ảnh crop nhỏ/chất lượng webcam), ta vẫn đưa NGUYÊN ảnh
    person_crop cho DeepFace - vì DeepFace tự detect mặt bên trong bằng
    RetinaFace (enforce_detection=False) và đã được kiểm chứng hoạt động
    đúng khi test độc lập. Trước đây code trả về None ngay khi insightface
    không thấy mặt, khiến DeepFace không bao giờ được gọi tới -> bỏ lỡ
    match dù index/DeepFace hoàn toàn đúng.

    Không dùng embedding của insightface để so sánh trực tiếp với index,
    vì index được build bằng DeepFace -> hai không gian embedding khác nhau.
    """

    def __init__(
        self,
        db_folder,
        index_file="faces.index",
        path_file="paths.pkl",
        threshold=0.62,
        det_ctx_id=-1,  # -1 = CPU (khớp với CUDA_VISIBLE_DEVICES=-1 ở trên)
    ):
        self.face_detector = ArcFaceExtractor(ctx_id=det_ctx_id)

        self.recognizer = ArcFaceRecognizer(
            db_folder=db_folder,
            index_file=index_file,
            path_file=path_file,
            threshold=threshold,
        )

        # Index đã build sẵn từ trước -> load thẳng, không build lại
        self.recognizer.load_index()

    def identify(self, person_crop):
        """
        person_crop: ảnh BGR (numpy array) của 1 người, đã được YOLO cắt ra.

        Trả về:
            (name, score)  nếu match được người trong database
            (None, score)  nếu có ảnh nhưng không match / dưới ngưỡng
            (None, 0.0)    nếu person_crop rỗng hoặc DeepFace lỗi hoàn toàn
        """
        if person_crop is None or person_crop.size == 0:
            return None, 0.0

        # Mặc định: dùng nguyên ảnh người, để DeepFace tự detect mặt bên trong
        face_crop = person_crop

        try:
            faces = self.face_detector.detect_faces(person_crop)
        except Exception as e:
            print(f"[FaceIdentifier] insightface detect lỗi, bỏ qua bước cắt gọn: {e}")
            faces = []

        if len(faces) > 0:
            # Lấy khuôn mặt lớn nhất trong crop (thường là rõ nét nhất)
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            x1, y1, x2, y2 = face.bbox.astype(int)
            h, w = person_crop.shape[:2]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            cropped = person_crop[y1:y2, x1:x2]

            if cropped.size > 0:
                face_crop = cropped
        else:
            print("[FaceIdentifier] insightface không thấy mặt trong crop -> "
                  "dùng nguyên ảnh người, để DeepFace tự detect")

        # Lưu tạm ra file trước khi đưa cho DeepFace (ổn định nhất khi nhận img_path
        # là đường dẫn, tránh phụ thuộc việc bản DeepFace đang cài có hỗ trợ ndarray)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)

        try:
            cv2.imwrite(tmp_path, face_crop)
            result = self.recognizer.search(tmp_path)
        except Exception as e:
            print(f"[FaceIdentifier] DeepFace search lỗi: {e}")
            return None, 0.0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if result["matched"]:
            name = os.path.splitext(os.path.basename(result["path"]))[0]
            return name, result["score"]

        return None, result["score"]

    def identify_from_candidates(self, crops):
        """
        crops: list các ảnh BGR (numpy array) ứng viên của CÙNG 1 người,
        thường được sắp xếp từ ảnh có bbox lớn nhất -> nhỏ nhất.

        Thử nhận diện lần lượt từng ảnh, DỪNG NGAY khi tìm được match đầu
        tiên (không cần thử hết). Nếu không ảnh nào match, trả về điểm cao
        nhất từng thấy được trong số các ảnh đã thử - để vẫn có con số hữu
        ích khi log/debug (thay vì chỉ mù mờ báo "unknown").
        """
        best_score = 0.0

        for crop in crops:
            name, score = self.identify(crop)

            if name:
                return name, score

            if score > best_score:
                best_score = score

        return None, best_score