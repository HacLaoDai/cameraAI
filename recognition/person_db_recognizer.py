import time
import threading

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

    QUAN TRỌNG (fix độ trễ 15-20s sau EXIT): bản trước gọi self.refresh()
    ĐỒNG BỘ (blocking) ngay trong identify() - mỗi khi đến hạn
    refresh_interval, cả pipeline nhận diện phải DỪNG LẠI để chờ query toàn
    bộ persons từ MongoDB qua mạng (WAN) xong mới tiếp tục detect mặt. Nếu
    query này chậm (mạng lag, DB tải cao...), toàn bộ hàng đợi track đang
    chờ xử lý bị kẹt theo. Giờ refresh() chạy trong 1 THREAD NỀN RIÊNG, độc
    lập hoàn toàn với luồng nhận diện - dù query Mongo có chậm cỡ nào cũng
    KHÔNG làm chậm việc detect/match của các track đang chờ trong queue.
    """

    def __init__(
        self,
        ctx_id=0,          # -1 = CPU, 0 = GPU đầu tiên
        threshold=0.45,     # ngưỡng cosine similarity để coi là match - cần tự test/tinh chỉnh
        refresh_interval=None,  # giây - tự động load lại persons từ DB sau mỗi khoảng này
        det_size=(256, 256),  # giảm từ mặc định (640,640) -> detect mặt nhanh hơn nhiều trên
                               # CPU, đặc biệt vì input đã là crop nhỏ (không cần độ phân giải lớn)
        auto_refresh_thread=True,  # True = refresh chạy nền định kỳ, không chặn identify()
        face_margin_ratio=0.3,  # thêm biên xung quanh bbox mặt khi crop để LƯU vào DB
                                 # (không ảnh hưởng embedding dùng để search/so khớp)
    ):
        self.face_extractor = ArcFaceExtractor(ctx_id=ctx_id, det_size=det_size)
        self.threshold = threshold
        self.refresh_interval = refresh_interval
        self.face_margin_ratio = face_margin_ratio

        self.index = None
        # row_meta[i] = thông tin người ứng với vector thứ i trong index
        # (1 người có thể có nhiều embedding -> nhiều dòng cùng person_id)
        self.row_meta = []

        self._refresh_lock = threading.Lock()
        self._last_refresh = 0

        # Load lần đầu - BẮT BUỘC chạy đồng bộ (blocking) vì cần có index
        # trước khi RecognitionWorker bắt đầu xử lý bất kỳ track nào.
        self.refresh(force=True)

        self._stop_refresh = threading.Event()
        self._refresh_thread = None

        if auto_refresh_thread:
            self._refresh_thread = threading.Thread(
                target=self._auto_refresh_loop, daemon=True, name="PersonDBRefresh"
            )
            self._refresh_thread.start()

    # ----------------------------------
    # Thread nền: tự refresh định kỳ, KHÔNG liên quan/không chặn identify()
    # ----------------------------------
    def _auto_refresh_loop(self):
        while not self._stop_refresh.is_set():
            if self._stop_refresh.wait(self.refresh_interval):
                break
            try:
                self.refresh(force=True)
            except Exception as e:
                print(f"[PersonDBRecognizer] Lỗi refresh nền (bỏ qua, giữ index cũ): {e}")

    def stop(self):
        """Gọi khi tắt chương trình để dừng thread refresh nền gọn gàng."""
        self._stop_refresh.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)

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
            with self._refresh_lock:
                self.index = None
                self.row_meta = []
                self._last_refresh = now
            return

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # Chỉ khoá lúc GÁN kết quả mới (rất nhanh) - phần build index/query
        # Mongo phía trên KHÔNG cần khoá, để identify() ở thread khác không
        # bao giờ phải chờ refresh() đang chạy dở.
        with self._refresh_lock:
            self.index = index
            self.row_meta = row_meta
            self._last_refresh = now

        print(
            f"[PersonDBRecognizer] Đã load {len(row_meta)} embedding / "
            f"{len(persons)} người từ MongoDB persons"
        )

    # ----------------------------------
    # Crop khuôn mặt (kèm margin) từ ảnh người, dùng để LƯU vào DB - tách
    # riêng khỏi ảnh dùng để search (search vẫn dùng bbox insightface trả
    # về nguyên bản, không cộng margin, để không ảnh hưởng embedding).
    # ----------------------------------
    def _crop_face_with_margin(self, person_crop, bbox):
        h, w = person_crop.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        mx = int(bw * self.face_margin_ratio)
        my = int(bh * self.face_margin_ratio)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        crop = person_crop[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop.copy()

    # ----------------------------------
    # Identify 1 ảnh
    # ----------------------------------
    def identify(self, person_crop):
        """
        person_crop: ảnh BGR (numpy array) của 1 người, đã được YOLO cắt ra.

        Trả về:
            ({"person_id":, "name":, "type":}, score, face_crop, embedding)  nếu match
            (None, score, face_crop, embedding)  nếu có mặt nhưng không match / dưới ngưỡng
            (None, 0.0, None, None)              nếu không tìm thấy khuôn mặt nào trong crop

        face_crop: ảnh khuôn mặt đã cắt riêng (kèm margin) từ person_crop -
        dùng để LƯU vào DB thay vì lưu nguyên ảnh toàn thân.

        embedding: vector khuôn mặt đã normalize L2 (numpy 1D) dùng để search
        - trả về LUÔN, kể cả khi không match ai trong DB persons, để nơi gọi
        (RecognitionWorker) có thể dùng nó tra/lưu vào "gallery người lạ"
        (UnknownGallery) - nhận ra người lạ quay lại sau X phút dù họ chưa
        từng được enroll thủ công.

        LƯU Ý: không còn gọi self.refresh() ở đây nữa (xem docstring class) -
        index được cập nhật bởi thread nền riêng, identify() chỉ ĐỌC
        self.index hiện có, không bao giờ bị chặn bởi việc query Mongo.
        """
        if person_crop is None or person_crop.size == 0:
            return None, 0.0, None, None

        with self._refresh_lock:
            index = self.index
            row_meta = self.row_meta

        t0 = time.time()
        embeddings, bboxes = self.face_extractor.extract_embeddings(person_crop)
        detect_ms = (time.time() - t0) * 1000

        if detect_ms > 500:
            print(f"[PersonDBRecognizer] [SLOW] insightface detect mất {detect_ms:.0f}ms "
                  f"cho 1 ảnh - cân nhắc giảm det_size hoặc giảm top_k")

        if len(embeddings) == 0:
            return None, 0.0, None, None

        # Lấy khuôn mặt có diện tích lớn nhất (thường rõ nét nhất)
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        best_idx = int(np.argmax(areas))

        face_crop = self._crop_face_with_margin(person_crop, bboxes[best_idx])

        query = np.array([embeddings[best_idx]], dtype=np.float32)
        faiss.normalize_L2(query)
        query_vec = query[0].copy()

        if index is None:
            # persons rỗng (chưa enroll ai) - KHÔNG search được, nhưng vẫn
            # trả về embedding để UnknownGallery hoạt động bình thường dù
            # DB known-person đang trống.
            return None, 0.0, face_crop, query_vec

        D, I = index.search(query, 1)

        score = float(D[0][0])
        idx = int(I[0][0])

        if idx == -1:
            return None, 0.0, face_crop, query_vec

        person = row_meta[idx]

        if score >= self.threshold:
            return person, score, face_crop, query_vec

        return None, max(score, 0.0), face_crop, query_vec

    # ----------------------------------
    # Identify nhiều ảnh ứng viên của CÙNG 1 người, dừng ngay khi match
    # ----------------------------------
    def identify_from_candidates(self, crops):
        """
        crops: list các ảnh BGR (numpy array) ứng viên của CÙNG 1 người,
        thường được sắp xếp từ ảnh có bbox lớn nhất -> nhỏ nhất.

        Trả về (person, score, face_crop, embedding):
            - Thử lần lượt từng ảnh, dừng ngay khi tìm được match đầu tiên.
            - Nếu không ảnh nào match, trả về điểm cao nhất từng thấy được.
            - face_crop là ảnh MẶT (không phải toàn thân) ứng với kết quả
              tốt nhất tìm được, kể cả khi unknown (miễn có phát hiện mặt).
            - embedding ứng với CHÍNH face_crop đó - dùng để tra/lưu vào
              UnknownGallery khi person=None (xem identify()).
        """
        t0 = time.time()
        best_score = 0.0
        best_face_crop = None
        best_embedding = None

        for crop in crops:
            person, score, face_crop, embedding = self.identify(crop)

            if face_crop is not None and (best_face_crop is None or score > best_score):
                best_face_crop = face_crop
                best_embedding = embedding

            if person:
                total_ms = (time.time() - t0) * 1000
                if total_ms > 1000:
                    print(f"[PersonDBRecognizer] [TIMING] identify_from_candidates "
                          f"mất {total_ms:.0f}ms ({len(crops)} candidates)")
                return person, score, face_crop, embedding

            if score > best_score:
                best_score = score

        total_ms = (time.time() - t0) * 1000
        if total_ms > 1000:
            print(f"[PersonDBRecognizer] [TIMING] identify_from_candidates "
                  f"mất {total_ms:.0f}ms ({len(crops)} candidates, unknown)")

        return None, best_score, best_face_crop, best_embedding