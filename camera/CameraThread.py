import os
import cv2
import threading
import time
from queue import Queue, Empty

# Ép OpenCV/FFmpeg dùng RTSP qua TCP thay vì UDP (mặc định). Log thực tế cho
# thấy rất nhiều lỗi giải mã HEVC kiểu "Could not find ref with POC",
# "Error constructing the frame RPS", "Unknown slice type" - đặc trưng của
# việc MẤT GÓI UDP giữa đường (rất phổ biến với RTSP qua Wi-Fi/mạng WAN, hoặc
# NVR có bitrate cao). TCP đổi lại vài ms độ trễ để đổi lấy stream không mất
# gói, ổn định hơn nhiều cho pipeline nhận diện. Phải set biến môi trường
# NÀY TRƯỚC khi cv2.VideoCapture() được gọi lần đầu tiên trong tiến trình.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
)


class CameraThread:
    """
    Tự động kết nối lại vô thời hạn trong CẢ 2 trường hợp:
      1. Camera bị rớt kết nối GIỮA CHỪNG (đang chạy bình thường rồi mất tín
         hiệu) - như bản gốc đã xử lý.
      2. Camera KHÔNG kết nối được NGAY TỪ LÚC KHỞI ĐỘNG (start() thất bại
         lần đầu) - bản gốc TRƯỚC ĐÂY sẽ bỏ cuộc luôn, không bao giờ thử lại
         nữa cho tới khi khởi động lại toàn bộ chương trình. Giờ cả 2 trường
         hợp đều dùng chung 1 vòng lặp retry chạy nền, không phân biệt.

    Backoff tăng dần (2s -> 4s -> 8s ... tối đa max_reconnect_interval) để
    tránh spam log/kết nối liên tục khi camera mất tín hiệu kéo dài (thay vì
    thử lại đúng mỗi 2s vô thời hạn như bản gốc).
    """

    def __init__(
        self,
        rtsp_url,
        queue_size=1,
        reconnect_interval=2,
        max_reconnect_interval=30,
    ):

        self.rtsp_url = rtsp_url

        # Camera
        self.cap = None

        # Frame Queue
        self.frame_queue = Queue(maxsize=queue_size)

        # Trạng thái
        self.running = False
        self.connected = False

        # Thread
        self.thread = None

        # Backoff cho việc reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self._current_backoff = reconnect_interval

        # Báo hiệu lần thử kết nối ĐẦU TIÊN đã xong (thành công hay thất
        # bại) - dùng để start() có thể trả về True/False như hành vi cũ,
        # trong khi vòng lặp reconnect nền vẫn tiếp tục chạy phía sau dù
        # lần đầu có thất bại.
        self._first_attempt_done = threading.Event()

    # ==========================
    # Mở VideoCapture (backend FFMPEG rõ ràng để option TCP có hiệu lực).
    # Với webcam local (rtsp_url là số int), backend FFMPEG không cần thiết
    # và có thể không hoạt động tốt trên mọi máy -> dùng backend mặc định.
    # ==========================
    def _open_capture(self):
        if isinstance(self.rtsp_url, str) and self.rtsp_url.lower().startswith("rtsp"):
            return cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        return cv2.VideoCapture(self.rtsp_url)

    def _try_connect(self, is_first_attempt=False):
        """Thử mở camera 1 lần. Trả về True/False. Không raise."""
        if self.cap is not None:
            self.cap.release()

        self.cap = self._open_capture()

        if self.cap.isOpened():
            self.connected = True
            self._current_backoff = self.reconnect_interval  # reset backoff
            if is_first_attempt:
                print("[INFO] Camera started.")
            else:
                print("[INFO] Reconnect success.")
            return True

        self.connected = False
        if is_first_attempt:
            print("[ERROR] Cannot connect camera.")
        else:
            print(f"[ERROR] Reconnect failed. Thử lại sau {self._current_backoff:.0f}s")
        return False

    # ==========================
    # Start Camera
    # ==========================
    def start(self):
        """
        Khởi động thread nền. Trả về True/False cho biết LẦN THỬ ĐẦU TIÊN có
        kết nối được hay không (để tương thích với code gọi cũ), nhưng dù
        kết quả là gì, thread nền vẫn tiếp tục tự động thử kết nối lại phía
        sau - người gọi KHÔNG cần tự retry start() nữa.
        """
        print("[INFO] Starting camera...")

        self.running = True
        self._first_attempt_done.clear()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="CameraThread"
        )
        self.thread.start()

        # Đợi đúng lần thử đầu tiên xong (không đợi toàn bộ quá trình retry)
        self._first_attempt_done.wait()

        return self.connected

    # ==========================
    # Stop Camera
    # ==========================
    def stop(self):

        print("[INFO] Stop camera...")

        self.running = False

        if self.thread is not None:
            self.thread.join()

        if self.cap is not None:
            self.cap.release()

        self.connected = False

        print("[INFO] Camera stopped.")

    # ==========================
    # Vòng lặp nền DUY NHẤT: vừa đọc frame, vừa tự kết nối/kết nối lại,
    # dùng chung cho cả lúc khởi động lẫn lúc mất kết nối giữa chừng.
    # ==========================
    def _run(self):
        is_first_attempt = True

        while self.running:

            if not self.connected:
                success = self._try_connect(is_first_attempt=is_first_attempt)

                if is_first_attempt:
                    is_first_attempt = False
                    self._first_attempt_done.set()

                if not success:
                    # Chờ theo backoff hiện tại, nhưng vẫn kiểm tra self.running
                    # mỗi giây để có thể dừng ngay khi stop() được gọi, không
                    # phải đợi hết nguyên khoảng backoff.
                    waited = 0
                    while waited < self._current_backoff and self.running:
                        time.sleep(1)
                        waited += 1

                    self._current_backoff = min(
                        self._current_backoff * 2, self.max_reconnect_interval
                    )
                    continue

            ret, frame = self.cap.read()

            if not ret:
                print("[WARNING] Camera disconnected.")
                self.connected = False
                continue

            # Queue chỉ giữ frame mới nhất
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass

            self.frame_queue.put_nowait(frame)

    # ==========================
    # Read newest frame
    # ==========================
    def read(self, timeout=0.1):

        try:

            frame = self.frame_queue.get(timeout=timeout)

            return True, frame

        except Empty:

            return False, None

    # ==========================
    # Camera status
    # ==========================
    def is_connected(self):

        return self.connected