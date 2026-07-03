import cv2
import time


class MotionDetector:

    def __init__(
        self,
        min_area=5000,
        cooldown=0,
        history=500,
        var_threshold=16,
    ):

        self.min_area = min_area
        self.cooldown = cooldown

        self.last_trigger = 0

        self.bg_subtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=False,
            )
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

    def detect(self, frame):

        mask = self.bg_subtractor.apply(frame)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_boxes = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            motion_boxes.append(
                (x, y, w, h)
            )

        motion = len(motion_boxes) > 0

        if motion:

            now = time.time()

            if (
                now - self.last_trigger
                < self.cooldown
            ):
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