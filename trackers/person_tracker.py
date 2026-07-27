import os
import cv2
import math
import threading
from datetime import datetime


class PersonTracker:
    """
    QUAN TRỌNG - THAY ĐỔI SO VỚI BẢN GỐC:
    Bản gốc chọn ảnh đại diện CHỈ theo diện tích bbox toàn thân (area) - dẫn
    tới việc lưu nhầm các khung hình chỉ thấy tay/lưng/vai (bbox to do người
    đứng gần camera) thay vì mặt, làm nhận diện khuôn mặt phía sau thất bại
    (luôn ra "Người lạ" dù người đó đã enroll).

    Bản này thêm 2 tiêu chí, ưu tiên theo thứ tự:
        1. has_face   - crop có phát hiện được khuôn mặt hay không (Haar
                         Cascade - rất nhẹ, KHÔNG dùng insightface ở đây vì
                         PersonTracker chạy ngay trong thread đọc camera,
                         không được phép chặn/làm chậm luồng đọc frame -
                         insightface vẫn chỉ chạy ở RecognitionWorker như cũ)
        2. sharpness  - độ nét (Laplacian variance) - loại ảnh bị mờ/motion
                         blur, ưu tiên khoảnh khắc người đứng yên/di chuyển
                         chậm hơn khi đi ngang camera
        3. area       - diện tích bbox (tiêu chí phụ, giữ như bản gốc)

    Nhờ vậy nếu track có ÍT NHẤT 1 khung hình bắt được mặt trong suốt vòng
    đời, khung đó luôn được ưu tiên chọn làm ảnh đại diện + ảnh gửi đi nhận
    diện, thay vì bị khung tay/lưng to hơn nhưng vô dụng lấn át.
    """

    # Cascade dùng chung cho mọi instance (load 1 lần, tránh tốn thời gian
    # load lại file XML mỗi khi tạo PersonTracker cho từng camera).
    #
    # _face_cascade_broken: nếu môi trường cv2 bị hỏng/xung đột nhiều gói
    # (opencv-python + opencv-python-headless cài chung -> thiếu hẳn class
    # CascadeClassifier, xem AttributeError), việc gọi lại cv2.CascadeClassifier
    # ở MỌI frame sẽ ném lỗi liên tục, làm chết hẳn thread camera (crash toàn
    # bộ pipeline như log thực tế đã xảy ra). Cờ này đảm bảo lỗi môi trường
    # chỉ được phát hiện + log CẢNH BÁO đúng 1 LẦN, sau đó tự động fallback
    # về chế độ KHÔNG kiểm tra "có mặt" (chỉ dùng sharpness + area) để
    # chương trình vẫn chạy tiếp, không crash - thay vì phải sửa xong môi
    # trường mới chạy được.
    _face_cascade = None
    _face_cascade_broken = False
    # QUAN TRỌNG - FIX SEGFAULT: _face_cascade là 1 object DÙNG CHUNG cho
    # MỌI camera (class-level, load 1 lần). Nhưng mỗi camera chạy trong 1
    # thread riêng (CameraPipeline trong multi_main.py) và _has_face() được
    # gọi ngay trong thread đó cho MỖI frame. Khi có từ 2 camera trở lên xử
    # lý người cùng lúc, 2 thread khác nhau gọi cascade.detectMultiScale()
    # ĐỒNG THỜI trên CÙNG 1 object cv2.CascadeClassifier - OpenCV KHÔNG đảm
    # bảo an toàn cho việc này (cascade dùng buffer nội bộ dùng chung giữa
    # các lần gọi), gây lỗi bộ nhớ ở tầng C++ -> Segmentation fault, không
    # bắt được bằng try/except ở Python. Lock này đảm bảo tại 1 thời điểm
    # chỉ 1 thread được dùng cascade - chi phí thêm không đáng kể vì
    # detectMultiScale trên 1 crop nhỏ vốn đã rất nhanh (vài ms).
    _face_cascade_lock = threading.Lock()

    @classmethod
    def _get_face_cascade(cls):
        if cls._face_cascade_broken:
            return None

        if cls._face_cascade is None:
            try:
                path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(path)
                if cascade.empty():
                    raise RuntimeError(f"Không load được file cascade: {path}")
                cls._face_cascade = cascade
            except (AttributeError, RuntimeError) as e:
                cls._face_cascade_broken = True
                print(
                    "[PersonTracker] [LOI MOI TRUONG cv2] Khong dung duoc "
                    "cv2.CascadeClassifier -> TAM BO QUA kiem tra 'co mat', "
                    "chi dung do net + dien tich de chon anh dai dien.\n"
                    f"    Chi tiet loi: {e}\n"
                    "    Nguyen nhan thuong gap: cai xung dot nhieu goi opencv "
                    "(vd opencv-python + opencv-python-headless cung luc).\n"
                    "    Cach sua: pip uninstall opencv-python opencv-python-headless "
                    "opencv-contrib-python opencv-contrib-python-headless -y "
                    "--break-system-packages && pip install opencv-python "
                    "--break-system-packages"
                )
                return None
        return cls._face_cascade

    def __init__(
        self,
        save_dir="saved_person",
        exit_timeout=10,
        on_person_saved=None,
        top_k=3,                 # tăng từ 1 -> 3: giữ nhiều ứng viên hơn để
                                  # identify_from_candidates() có cái để thử,
                                  # thay vì chỉ có đúng 1 lựa chọn duy nhất
        merge_window=30,        # số frame giữ lại track vừa EXIT để chờ merge (vd 30 frame ~ 1-1.5s @ 20-30fps)
        merge_max_dist_ratio=0.25,  # khoảng cách bbox center tối đa (tỉ lệ theo đường chéo khung hình) để coi là cùng người
        min_crop_width=60,      # bbox hẹp hơn mức này (px) gần như chắc chắn KHÔNG đủ chi tiết để
                                 # insightface tìm ra mặt (vd chỉ thấy 1 dải tóc/tay do người bị che/cắt
                                 # ở mép khung hình) -> loại khỏi danh sách ứng viên nhận diện
        max_aspect_ratio=3.5,   # bbox có h/w vượt ngưỡng này (quá cao & hẹp bất thường so với 1 người
                                 # đứng bình thường ~1.5-2.5) thường là dấu hiệu crop bị cắt/che một phần
        diversity_sample_interval=5,  # cứ mỗi N frame lại lấy thêm 1 crop "đa dạng" (KHÔNG chỉ chọn
                                 # theo diện tích lớn nhất) để tăng cơ hội bắt được khoảnh khắc người
                                 # quay mặt về camera, thay vì luôn là frame lúc họ quay đi/giơ tay che mặt
                                 # (giảm từ 15 -> 5: track thường chỉ tồn tại 1-2s khi người đi ngang
                                 # camera, 15 frame gần như là cả vòng đời track -> quá thưa)
        diversity_pool_size=5,   # tăng từ 2 -> 5: giữ nhiều mẫu hơn để không bỏ lỡ khoảnh khắc mặt
        min_sharpness=40,        # ngưỡng Laplacian variance - crop dưới mức này bị coi là quá mờ
                                  # để dùng làm ảnh đại diện/nhận diện (CẦN tự test lại với camera
                                  # thực tế của bạn, xem ghi chú ở cuối file)
        min_face_size=(24, 24),  # kích thước tối thiểu (px) để Haar Cascade coi là 1 khuôn mặt hợp lệ
    ):
        self.save_dir = save_dir
        self.exit_timeout = exit_timeout
        self.top_k = top_k
        self.merge_window = merge_window
        self.merge_max_dist_ratio = merge_max_dist_ratio
        self.min_crop_width = min_crop_width
        self.max_aspect_ratio = max_aspect_ratio
        self.diversity_sample_interval = diversity_sample_interval
        self.diversity_pool_size = diversity_pool_size
        self.min_sharpness = min_sharpness
        self.min_face_size = min_face_size

        # Callback: được gọi với (track_id, crops) ngay sau khi 1 người EXIT thật sự
        # (đã hết grace period mà không có track nào merge vào).
        self.on_person_saved = on_person_saved
        os.makedirs(save_dir, exist_ok=True)

        self.tracks = {}
        # Các track vừa EXIT, đang trong thời gian chờ để merge với track mới
        # key: old_track_id -> {"info":..., "exit_frame":..., "last_bbox":...}
        self.pending_exits = {}

    def _is_plausible_crop(self, x1, y1, x2, y2):
        """Lọc sơ bộ các bbox gần như chắc chắn không hữu ích cho nhận diện
        mặt (quá hẹp / tỉ lệ bất thường do người bị che hoặc cắt ở mép khung
        hình) - KHÔNG loại hoàn toàn khỏi tracking, chỉ loại khỏi việc được
        chọn làm ảnh đại diện để nhận diện."""
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False
        if w < self.min_crop_width:
            return False
        if (h / w) > self.max_aspect_ratio:
            return False
        return True

    # ------------------------------------------------------------
    # Đo chất lượng crop: có mặt hay không + độ nét
    # ------------------------------------------------------------
    def _has_face(self, crop_img):
        """Kiểm tra nhanh crop có khuôn mặt hay không bằng Haar Cascade.
        Cố tình dùng cascade thay vì insightface: cascade chạy được vài ms
        trên CPU, phù hợp để gọi mỗi frame trong thread đọc camera, trong
        khi insightface (900ms-1s/lần nếu không giới hạn module) chỉ nên
        chạy ở RecognitionWorker (thread nền riêng) như thiết kế hiện tại."""
        cascade = self._get_face_cascade()
        if cascade is None:
            # Môi trường cv2 bị hỏng (xem _get_face_cascade) -> không xác
            # định được có mặt hay không. Trả về False (không ưu tiên hơn
            # crop khác), pipeline vẫn chạy tiếp bình thường chỉ dựa vào
            # sharpness + area, KHÔNG throw để không làm chết thread camera.
            return False

        try:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            with self._face_cascade_lock:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=self.min_face_size,
                )
            return len(faces) > 0
        except cv2.error:
            return False

    def _compute_sharpness(self, crop_img):
        """Laplacian variance - điểm càng cao càng nét. Dùng để loại ảnh bị
        mờ/motion blur (vd người đang bước nhanh qua camera)."""
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def update(self, detections, frame, frame_index):
        current_ids = set()
        h, w = frame.shape[:2]
        diag = math.hypot(w, h)

        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            current_ids.add(track_id)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            area = (x2 - x1) * (y2 - y1)
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            if track_id not in self.tracks:
                merged_from = self._try_merge(track_id, center, frame_index, diag)
                if merged_from is not None:
                    print(f"[MERGE] {track_id} <- {merged_from} (cùng 1 người)")
                else:
                    self.tracks[track_id] = {
                        "enter_time": datetime.now(),
                        "last_frame": frame_index,
                        "last_seen_time": datetime.now(),
                        "top_crops": [],
                        "diversity_crops": [],
                        "last_diversity_frame": frame_index,
                        "saved": False,
                    }
                    print(f"[ENTER] {track_id}")

            track = self.tracks[track_id]
            track["last_frame"] = frame_index
            track["last_bbox_center"] = center
            # Moc thoi gian THUC (wall-clock) nguoi nay CON DUOC THAY o
            # camera nay - dung de doi chieu cheo camera (xem
            # multi_main.py RecognitionWorker._resolve_cross_camera):
            # camera nao nguoi do BIEN MAT (last_seen_time nho hon) TRUOC
            # moi la huong camera dung thuc te khi 2 camera doi dien nhau
            # cung thay 1 nguoi. Dung datetime.now() (dong ho thuc) thay vi
            # frame_index vi frame_index KHONG the so sanh giua 2 camera
            # khac nhau (FPS/do tre khac nhau moi camera).
            track["last_seen_time"] = datetime.now()

            if area > 0:
                plausible = self._is_plausible_crop(x1, y1, x2, y2)

                if plausible:
                    crop_img = frame[y1:y2, x1:x2].copy()
                    self._update_top_crops(track, area, crop_img)
                    self._maybe_add_diversity_sample(track, crop_img, frame_index)

        self._handle_exits(current_ids, frame_index)
        self._flush_expired_pending(frame_index)
        return current_ids

    def _try_merge(self, new_track_id, new_center, frame_index, diag):
        """Tìm track vừa EXIT gần đây, ở vị trí gần track mới -> merge thành 1 người."""
        best_match = None
        best_dist = None

        for old_id, entry in self.pending_exits.items():
            if frame_index - entry["exit_frame"] > self.merge_window:
                continue
            old_center = entry["last_bbox_center"]
            dist = math.hypot(
                new_center[0] - old_center[0],
                new_center[1] - old_center[1],
            )
            if dist <= diag * self.merge_max_dist_ratio:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_match = old_id

        if best_match is None:
            return None

        # Gộp: track mới kế thừa dữ liệu track cũ (giữ nguyên top_crops, chưa saved)
        entry = self.pending_exits.pop(best_match)
        self.tracks[new_track_id] = entry["info"]
        return best_match

    def _update_top_crops(self, track, area, crop_img):
        """Sắp xếp ứng viên theo thứ tự ưu tiên: có mặt > độ nét > diện tích.
        Nhờ vậy 1 crop CÓ mặt (dù diện tích nhỏ hơn) luôn được xếp trên 1
        crop KHÔNG có mặt (dù to hơn nhiều, vd tay/lưng đưa sát camera)."""
        has_face = self._has_face(crop_img)
        sharpness = self._compute_sharpness(crop_img)

        top_crops = track["top_crops"]
        top_crops.append((has_face, sharpness, area, crop_img))
        top_crops.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        if len(top_crops) > self.top_k:
            del top_crops[self.top_k:]

    def _maybe_add_diversity_sample(self, track, crop_img, frame_index):
        """Bên cạnh top-k theo (có mặt, độ nét, diện tích), định kỳ lấy
        thêm 1 mẫu bất kể điểm số để tăng cơ hội bắt được khoảnh khắc mặt
        quay về camera trong suốt vòng đời của track."""
        if (frame_index - track["last_diversity_frame"]) < self.diversity_sample_interval:
            return

        track["last_diversity_frame"] = frame_index
        pool = track["diversity_crops"]
        pool.append(crop_img)
        if len(pool) > self.diversity_pool_size:
            pool.pop(0)  # giữ các mẫu MỚI NHẤT, bỏ mẫu cũ nhất

    def _candidate_crops(self, info):
        """Gộp candidate từ top_crops (có mặt + nét + diện tích) +
        diversity_crops (đa dạng theo thời gian) thành 1 danh sách duy nhất
        để thử nhận diện. top_crops đi trước vì đã được xếp hạng chất lượng
        tốt hơn -> identify_from_candidates() (dừng ngay khi match) sẽ ưu
        tiên thử các ứng viên tốt nhất trước."""
        crops = [c for _, _, _, c in info["top_crops"] if c.size > 0]
        crops += [c for c in info["diversity_crops"] if c.size > 0]
        return crops

    def _best_crop_for_saving(self, info):
        """Chọn ảnh NÉT NHẤT + CÓ MẶT (nếu có) để lưu làm ảnh đại diện.
        Nếu không track nào có mặt, fallback về ảnh nét nhất/diện tích lớn
        nhất trong top_crops (giữ hành vi tương tự bản gốc)."""
        if not info["top_crops"]:
            return None

        # top_crops đã được sort theo (has_face, sharpness, area) giảm dần
        # -> phần tử đầu tiên luôn là lựa chọn tốt nhất hiện có.
        _, _, _, best_crop = info["top_crops"][0]
        return best_crop

    def _handle_exits(self, current_ids, frame_index):
        lost_ids = []
        for track_id, info in self.tracks.items():
            if track_id in current_ids:
                continue
            if frame_index - info["last_frame"] > self.exit_timeout:
                print(f"[EXIT] {track_id}")
                # Không lưu/gọi callback ngay -> đưa vào pending_exits chờ merge
                self.pending_exits[track_id] = {
                    "info": info,
                    "exit_frame": frame_index,
                    "last_bbox_center": info.get("last_bbox_center", (0, 0)),
                }
                lost_ids.append(track_id)
        for track_id in lost_ids:
            del self.tracks[track_id]

    def _flush_expired_pending(self, frame_index):
        """Track nào chờ quá merge_window mà không ai merge vào -> coi là exit thật, lưu lại."""
        expired = []
        for old_id, entry in self.pending_exits.items():
            if frame_index - entry["exit_frame"] > self.merge_window:
                expired.append(old_id)

        for old_id in expired:
            entry = self.pending_exits.pop(old_id)
            self._save_crop(old_id, entry["info"])

    def _save_crop(self, track_id, info):
        if info["saved"] or not info["top_crops"]:
            return

        best_crop = self._best_crop_for_saving(info)
        if best_crop is None or best_crop.size == 0:
            return

        filename = os.path.join(self.save_dir, f"person_{track_id}.jpg")
        cv2.imwrite(filename, best_crop)
        info["saved"] = True
        if self.on_person_saved is not None:
            crops = self._candidate_crops(info)
            # last_seen_time = moc thoi gian nguoi nay THUC SU bien mat
            # khoi camera nay (frame cuoi cung con thay ho), KHONG PHAI
            # luc track duoc "chot" (co the tre hon vai giay do
            # exit_timeout + merge_window). Day chinh la gia tri can de
            # RecognitionWorker so sanh "camera nao bien mat truoc".
            disappear_time = info.get("last_seen_time")
            self.on_person_saved(track_id, crops, disappear_time)

    def finalize(self):
        # Xử lý các track đang active
        for track_id, info in list(self.tracks.items()):
            self._save_crop(track_id, info)
        # Xử lý cả các track đang trong pending_exits (chưa merge xong)
        for track_id, entry in list(self.pending_exits.items()):
            self._save_crop(track_id, entry["info"])
        self.pending_exits.clear()


# ======================================================
# GHI CHÚ - CÁCH TINH CHỈNH min_sharpness CHO CAMERA THỰC TẾ
# ======================================================
# Giá trị Laplacian variance phụ thuộc độ phân giải, texture, ánh sáng của
# từng camera - KHÔNG có 1 con số chuẩn dùng chung cho mọi hệ thống. Chạy
# đoạn dưới với vài ảnh mẫu đã biết "nét" và "mờ" lấy từ saved_person/ để
# chọn ngưỡng phù hợp:
#
#   import cv2
#   def sharpness(path):
#       img = cv2.imread(path)
#       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#       return cv2.Laplacian(gray, cv2.CV_64F).var()
#
#   print("Ảnh nét:", sharpness("saved_person/anh_ro.jpg"))
#   print("Ảnh mờ :", sharpness("saved_person/anh_mo.jpg"))
#   # chọn min_sharpness nằm giữa 2 giá trị này