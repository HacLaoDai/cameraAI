import time

import faiss
import numpy as np

from detectors.detect_face import ArcFaceExtractor
from database import task_db


class PersonDBRecognizer:
    """
    Nhận diện khuôn mặt = insightface (ArcFace) để detect + trích embedding,
    ghép với faiss để search nhanh trong toàn bộ `persons` (MongoDB).

    Lý do dùng insightface làm nguồn embedding (thay vì DeepFace):
        - `manage_persons.py` (công cụ enroll người) đang trích embedding
          bằng insightface (ArcFaceExtractor) rồi lưu vào persons.embeddings.
          Muốn search đúng, bên tìm kiếm BẮT BUỘC phải dùng cùng 1 model ->
          insightface. Nếu dùng DeepFace để search thì 2 không gian embedding
          khác nhau, so khớp sẽ sai dù điểm số vẫn có vẻ hợp lý.
        - insightface (ONNX runtime) nhẹ và ổn định hơn DeepFace+TensorFlow
          khi chạy CPU liên tục nhiều camera.

    faiss chỉ đóng vai trò tăng tốc bước search (so với vòng lặp numpy thủ
    công) khi số người trong DB lớn - không đổi kết quả, chỉ đổi tốc độ.
    """

    def __init__(
        self,
        ctx_id=-1,          # -1 = CPU, 0 = GPU đầu tiên
        threshold=0.45,     # ngưỡng cosine similarity để coi là match - cần tự test/tinh chỉnh
        refresh_interval=60,  # giây - tự động load lại persons từ DB sau mỗi khoảng này
    ):
        self.face_extractor = ArcFaceExtractor(ctx_id=ctx_id)
        self.threshold = threshold
        self.refresh_interval = refresh_interval

        self.index = None
        # row_meta[i] = thông tin người ứng với vector thứ i trong index
        # (1 người có thể có nhiều embedding -> nhiều dòng cùng person_id)
        self.row_meta = []

        self._last_refresh = 0
        self.refresh(force=True)

    # ----------------------------------
    # Build / refresh faiss index từ MongoDB
    # ----------------------------------
    def refresh(self, force=False):
        now = time.time()

        if not force and (now - self._last_refresh) < self.refresh_interval:
            return

        persons = task_db.get_all_persons()

        embeddings = []
        row_meta = []

        for p in persons:
            for emb in p.get("embeddings", []):
                embeddings.append(emb)
                row_meta.append({
                    "person_id": p["person_id"],
                    "name": p["name"],
                    "type": p.get("type"),
                })

        if len(embeddings) == 0:
            print("[PersonDBRecognizer] Không có embedding nào trong DB (persons rỗng).")
            self.index = None
            self.row_meta = []
            self._last_refresh = now
            return

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.row_meta = row_meta
        self._last_refresh = now

        print(
            f"[PersonDBRecognizer] Đã load {len(row_meta)} embedding / "
            f"{len(persons)} người từ MongoDB persons"
        )

    # ----------------------------------
    # Identify 1 ảnh
    # ----------------------------------
    def identify(self, person_crop):
        """
        person_crop: ảnh BGR (numpy array) của 1 người, đã được YOLO cắt ra.

        Trả về:
            ({"person_id":, "name":, "type":}, score)  nếu match
            (None, score)  nếu có mặt nhưng không match / dưới ngưỡng
            (None, 0.0)    nếu không tìm thấy khuôn mặt nào trong crop
        """
        self.refresh()

        if person_crop is None or person_crop.size == 0:
            return None, 0.0

        if self.index is None:
            return None, 0.0

        embeddings, bboxes = self.face_extractor.extract_embeddings(person_crop)

        if len(embeddings) == 0:
            return None, 0.0

        # Lấy khuôn mặt có diện tích lớn nhất (thường rõ nét nhất)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best_idx = int(np.argmax(areas))

        query = np.array([embeddings[best_idx]], dtype=np.float32)
        faiss.normalize_L2(query)

        D, I = self.index.search(query, 1)

        score = float(D[0][0])
        idx = int(I[0][0])

        if idx == -1:
            return None, 0.0

        person = self.row_meta[idx]

        if score >= self.threshold:
            return person, score

        return None, max(score, 0.0)

    # ----------------------------------
    # Identify nhiều ảnh ứng viên của CÙNG 1 người, dừng ngay khi match
    # ----------------------------------
    def identify_from_candidates(self, crops):
        """
        crops: list các ảnh BGR (numpy array) ứng viên của CÙNG 1 người,
        thường được sắp xếp từ ảnh có bbox lớn nhất -> nhỏ nhất.

        Trả về (person, score) giống identify(), nhưng thử lần lượt từng
        ảnh và dừng ngay khi tìm được match đầu tiên. Nếu không ảnh nào
        match, trả về điểm cao nhất từng thấy được (hữu ích khi log/debug).
        """
        best_score = 0.0

        for crop in crops:
            person, score = self.identify(crop)

            if person:
                return person, score

            if score > best_score:
                best_score = score

        return None, best_score