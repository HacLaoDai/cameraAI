import cv2
import numpy as np
import math

from camera.CameraThread import CameraThread


# Danh sách camera
RTSP_URLS = [
    "rtsp://localhost:8554/live",
    "rtsp://localhost:8554/live4",
    "rtsp://localhost:8554/live5",
    # "rtsp://localhost:8554/cam4",
]

def build_grid(frames, cell_width=320, cell_height=240):

    n = len(frames)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

    images = []

    for frame in frames:

        if frame is None:
            images.append(blank.copy())
        else:
            images.append(cv2.resize(frame, (cell_width, cell_height)))

    while len(images) < rows * cols:
        images.append(blank.copy())

    rows_img = []

    for r in range(rows):

        start = r * cols
        end = start + cols

        rows_img.append(cv2.hconcat(images[start:end]))

    grid = cv2.vconcat(rows_img)

    return grid



def main():

    cameras = []

    # Khởi tạo camera
    for url in RTSP_URLS:

        cam = CameraThread(url)

        if cam.start():
            cameras.append(cam)
        else:
            print(f"Không mở được {url}")

    if len(cameras) == 0:
        print("Không có camera nào hoạt động.")
        return

    while True:

        frames = []

        for cam in cameras:

            ret, frame = cam.read()

            if ret:
                frames.append(frame)
            else:
                frames.append(None)

        grid = build_grid(frames)

        cv2.imshow("Camera Monitor", grid)

        if cv2.waitKey(1) == 27:
            break

    for cam in cameras:
        cam.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()