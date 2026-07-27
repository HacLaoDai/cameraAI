import time
import threading
import uuid

import numpy as np


class UnknownGallery:
    """
    "Bộ nhớ tạm" cho người LẠ (chưa enroll trong DB `persons`) - giải quyết
    vấn đề: hiện tại nếu 1 người lạ rời khỏi camera rồi quay lại sau vài
    phút / vài giờ, hệ thống coi đây là 1 người lạ HOÀN TOÀN MỚI - không có
    danh tính ổn định nào để dedup theo hướng hoặc để đối chiếu "đây có phải
    người mà camera đối diện vừa thấy không".

    KHÁC với PersonDBRecognizer (search trong MongoDB `persons` đã enroll
    thủ công qua manage_persons.py): gallery này CHỈ sống trong RAM, tự
    động cấp 1 "temp_id" mới cho mỗi người lạ CHƯA TỪNG gặp, và trả lại
    ĐÚNG temp_id cũ nếu embedding khớp với người lạ đã gặp trước đó (trong
    vòng ttl_seconds gần nhất).

    KHÔNG đổi schema MongoDB - face_events vẫn lưu person_id=-1 cho mọi
    người lạ như cũ (xem task_db.save_face_to_db). temp_id ở đây chỉ dùng
    làm "khóa nhận dạng" NỘI BỘ trong RecognitionWorker (multi_main.py) để:
        1. Dedup theo hướng (is_in) cho người lạ - tránh spam DB khi 1
           người lạ đứng lảng vảng trước camera bị tách thành nhiều track.
        2. Đối chiếu 2 camera đối diện nhau (in/out) cùng thấy 1 người lạ
           CÙNG LÚC (xem RecognitionWorker._resolve_cross_camera) - nếu
           không có 1 danh tính ổn định, hệ thống không thể biết 2 sự kiện
           từ 2 camera là CÙNG một người lạ để so sánh ai biến mất trước.

    Threshold nên dùng CHUNG giá trị với PersonDBRecognizer.threshold (cùng
    không gian embedding insightface) trừ khi có lý do cụ thể để tách riêng.
    """

    def __init__(self, threshold=0.45, ttl_seconds=6 * 3600, cleanup_interval=300):
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        # RecognitionWorker chỉ có 1 thread duy nhất gọi identify_or_register(),
        # nhưng vẫn khoá vì thread cleanup nền cũng đụng vào self._entries.
        self._lock = threading.Lock()
        self._entries = []  # [{"temp_id":, "embedding": np.array(D,), "last_seen": float}]

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="UnknownGalleryCleanup"
        )
        self._thread.start()

    def stop(self):
        """Gọi khi tắt chương trình để dừng thread cleanup nền gọn gàng."""
        self._stop.set()
        self._thread.join(timeout=5)

    def _cleanup_loop(self):
        while not self._stop.wait(self.cleanup_interval):
            self._cleanup()

    def _cleanup(self):
        now = time.time()
        with self._lock:
            before = len(self._entries)
            self._entries = [
                e for e in self._entries if (now - e["last_seen"]) <= self.ttl_seconds
            ]
            removed = before - len(self._entries)
        if removed:
            print(f"[UnknownGallery] Dọn {removed} người lạ quá hạn "
                  f"({self.ttl_seconds}s không quay lại)")

    def identify_or_register(self, embedding):
        """
        embedding: numpy 1D, ĐÃ normalize L2 (đúng định dạng embedding trả
        về từ PersonDBRecognizer.identify()/identify_from_candidates()).

        Trả về temp_id (str):
            - temp_id CŨ nếu khớp 1 người lạ đã gặp trước đó (>= threshold).
            - temp_id MỚI nếu chưa từng gặp / embedding=None -> None.
        """
        if embedding is None:
            return None

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        now = time.time()

        with self._lock:
            best_id = None
            best_score = -1.0
            best_entry = None

            for entry in self._entries:
                score = float(np.dot(emb, entry["embedding"]))
                if score > best_score:
                    best_score = score
                    best_id = entry["temp_id"]
                    best_entry = entry

            if best_entry is not None and best_score >= self.threshold:
                best_entry["last_seen"] = now
                return best_id

            new_id = "unk_" + uuid.uuid4().hex[:10]
            self._entries.append({
                "temp_id": new_id,
                "embedding": emb,
                "last_seen": now,
            })
            return new_id