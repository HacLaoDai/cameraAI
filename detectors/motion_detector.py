import cv2
import time


class MotionDetector:
    def __init__(
        self,
        min_area=800,        # giảm từ 5000 -> phát hiện chuyển động nhỏ hơn
        cooldown=0,
        history=300,         # giảm để nền thích ứng nhanh hơn với thay đổi
        var_threshold=8,     # giảm từ 16 -> nhạy hơn với sai khác pixel nhỏ
    ):
        self.min_area = min_area
        self.cooldown = cooldown
        self.last_trigger = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # nhỏ hơn -> ít xóa mất vùng nhỏ
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def detect(self, frame):
        mask = self.bg_subtractor.apply(frame)

        # OPEN nhẹ để lọc nhiễu li ti, nhưng không quá mạnh
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # DILATE để nối các vùng chuyển động nhỏ/rời rạc thành khối lớn hơn min_area
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))

        motion = len(motion_boxes) > 0
        if motion:
            now = time.time()
            if now - self.last_trigger < self.cooldown:
                return False, []
            self.last_trigger = now
            return True, motion_boxes

        return False, []

# cap = cv2.VideoCapture(0)

# detector = MotionDetector(
#     min_area=4000,
#     cooldown=2
# )

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     motion, boxes = detector.detect(frame)

#     if motion:

#         print("Motion Detected!")

#         for x, y, w, h in boxes:

#             cv2.rectangle(
#                 frame,
#                 (x, y),
#                 (x + w, y + h),
#                 (0, 255, 0),
#                 2
#             )

#     cv2.imshow("Motion", frame)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()