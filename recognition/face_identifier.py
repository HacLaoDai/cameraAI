import os
import tempfile

import cv2

# Import build_system_regconition TRƯỚC để biến môi trường CUDA_VISIBLE_DEVICES=-1
# được set trước khi insightface/onnxruntime khởi tạo session (tránh xung đột GPU/CPU).
from recognition.build_system_regconition import ArcFaceRecognizer
from detectors.detect_face import ArcFaceExtractor


class FaceIdentifier:
    """
    Bước 1: dùng insightface (ArcFaceExtractor) chỉ để TÌM vị trí khuôn mặt
            trong ảnh người đã được YOLO cắt ra.
    Bước 2: cắt riêng khuôn mặt đó, đưa cho ArcFaceRecognizer (DeepFace + faiss)
            để so khớp với index đã build sẵn.

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
            (None, score)  nếu có mặt nhưng không match / dưới ngưỡng
            (None, 0.0)    nếu không tìm thấy khuôn mặt nào trong crop
        """
        if person_crop is None or person_crop.size == 0:
            return None, 0.0

        faces = self.face_detector.detect_faces(person_crop)

        if len(faces) == 0:
            return None, 0.0

        # Lấy khuôn mặt lớn nhất trong crop (thường là rõ nét nhất)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = person_crop.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = person_crop[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, 0.0

        # Lưu tạm ra file trước khi đưa cho DeepFace (ổn định nhất khi nhận img_path
        # là đường dẫn, tránh phụ thuộc việc bản DeepFace đang cài có hỗ trợ ndarray)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(tmp_fd)

        try:
            cv2.imwrite(tmp_path, face_crop)
            result = self.recognizer.search(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if result["matched"]:
            name = os.path.splitext(os.path.basename(result["path"]))[0]
            return name, result["score"]

        return None, result["score"]