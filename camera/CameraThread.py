import cv2
import threading
import time
from queue import Queue, Empty


class CameraThread:

    def __init__(self, rtsp_url, queue_size=1):

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

    # ==========================
    # Start Camera
    # ==========================
    def start(self):

        print("[INFO] Starting camera...")

        self.running = True

        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():

            print("[ERROR] Cannot connect camera.")

            self.connected = False
            return False

        self.connected = True

        self.thread = threading.Thread(
            target=self.update,
            daemon=True,
            name="CameraThread"
        )

        self.thread.start()

        print("[INFO] Camera started.")

        return True

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
    # Reconnect
    # ==========================
    def reconnect(self):

        print("[WARNING] Camera disconnected.")

        self.connected = False

        if self.cap is not None:
            self.cap.release()

        while self.running:

            print("[INFO] Reconnecting...")

            self.cap = cv2.VideoCapture(self.rtsp_url)

            if self.cap.isOpened():

                print("[INFO] Reconnect success.")

                self.connected = True

                return

            print("[ERROR] Reconnect failed.")

            time.sleep(2)

    # ==========================
    # Camera Thread
    # ==========================
    def update(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:

                self.reconnect()

                continue

            # Queue chỉ giữ frame mới nhất
            if self.frame_queue.full():

                self.frame_queue.get_nowait()

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