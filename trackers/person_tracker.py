import os

import cv2

from datetime import datetime


class PersonTracker:

    def __init__(
        self,
        save_dir="saved_person",
        exit_timeout=30,
        on_person_saved=None,
        top_k=3,
    ):

        self.save_dir = save_dir
        self.exit_timeout = exit_timeout
        self.top_k = top_k

        # Callback: được gọi với (track_id, crops) ngay sau khi 1 người EXIT.
        # crops là LIST các ảnh (numpy array), sắp xếp từ lớn -> nhỏ, không
        # phải 1 ảnh duy nhất - để nơi gọi (vd nhận diện khuôn mặt) có thể
        # thử lần lượt nhiều ảnh, tăng cơ hội gặp khung hình có mặt rõ.
        self.on_person_saved = on_person_saved

        os.makedirs(save_dir, exist_ok=True)

        self.tracks = {}

    def update(self, detections, frame, frame_index):

        current_ids = set()
        h, w = frame.shape[:2]

        for det in detections:

            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]

            current_ids.add(track_id)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            area = (x2 - x1) * (y2 - y1)

            if track_id not in self.tracks:

                self.tracks[track_id] = {
                    "enter_time": datetime.now(),
                    "last_frame": frame_index,
                    "top_crops": [],  # list [(area, crop_img), ...] sắp xếp giảm dần
                    "saved": False,
                }

                print(f"[ENTER] {track_id}")

            track = self.tracks[track_id]

            track["last_frame"] = frame_index

            if area > 0:
                crop_img = frame[y1:y2, x1:x2].copy()
                self._update_top_crops(track, area, crop_img)

        self._handle_exits(current_ids, frame_index)

        return current_ids

    def _update_top_crops(self, track, area, crop_img):
        top_crops = track["top_crops"]
        top_crops.append((area, crop_img))
        top_crops.sort(key=lambda t: t[0], reverse=True)

        if len(top_crops) > self.top_k:
            del top_crops[self.top_k:]

    def _handle_exits(self, current_ids, frame_index):

        lost_ids = []

        for track_id, info in self.tracks.items():

            if track_id in current_ids:
                continue

            if frame_index - info["last_frame"] > self.exit_timeout:

                print(f"[EXIT] {track_id}")

                self._save_crop(track_id, info)

                lost_ids.append(track_id)

        for track_id in lost_ids:

            del self.tracks[track_id]

    def _save_crop(self, track_id, info):

        if info["saved"] or not info["top_crops"]:
            return

        # Ảnh lớn nhất trong top_k -> lưu ra file (giữ nguyên hành vi cũ)
        _, best_crop = info["top_crops"][0]

        if best_crop.size == 0:
            return

        filename = os.path.join(self.save_dir, f"person_{track_id}.jpg")

        cv2.imwrite(filename, best_crop)

        info["saved"] = True

        if self.on_person_saved is not None:
            crops = [c for _, c in info["top_crops"] if c.size > 0]
            self.on_person_saved(track_id, crops)

    def finalize(self):

        for track_id, info in list(self.tracks.items()):

            self._save_crop(track_id, info)