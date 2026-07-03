import os

import cv2

from datetime import datetime


class PersonTracker:

    def __init__(
        self,
        save_dir="saved_person",
        exit_timeout=30,
    ):

        self.save_dir = save_dir
        self.exit_timeout = exit_timeout

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        self.tracks = {}

    def update(
        self,
        detections,
        frame,
        frame_index,
    ):

        current_ids = set()

        for det in detections:

            track_id = det["track_id"]

            x1, y1, x2, y2 = det["bbox"]

            current_ids.add(track_id)

            area = (
                (x2 - x1)
                * (y2 - y1)
            )

            if track_id not in self.tracks:

                self.tracks[track_id] = {
                    "enter_time": datetime.now(),
                    "last_frame": frame_index,
                    "best_area": 0,
                    "best_frame": None,
                    "best_box": None,
                    "saved": False,
                }

                print(
                    f"[ENTER] {track_id}"
                )

            track = self.tracks[
                track_id
            ]

            track["last_frame"] = (
                frame_index
            )

            if area > track["best_area"]:

                track["best_area"] = area

                track[
                    "best_frame"
                ] = frame.copy()

                track[
                    "best_box"
                ] = (x1, y1, x2, y2)

        self._handle_exits(
            current_ids,
            frame_index,
        )

        return current_ids

    def _handle_exits(
        self,
        current_ids,
        frame_index,
    ):

        lost_ids = []

        for (
            track_id,
            info,
        ) in self.tracks.items():

            if track_id in current_ids:
                continue

            if (
                frame_index
                - info["last_frame"]
                > self.exit_timeout
            ):

                print(
                    f"[EXIT] {track_id}"
                )

                self._save_crop(
                    track_id,
                    info,
                )

                lost_ids.append(
                    track_id
                )

        for track_id in lost_ids:

            del self.tracks[
                track_id
            ]

    def _save_crop(
        self,
        track_id,
        info,
    ):

        if (
            info["saved"]
            or info["best_frame"]
            is None
        ):
            return

        x1, y1, x2, y2 = (
            info["best_box"]
        )

        img = info[
            "best_frame"
        ]

        crop = img[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            return

        filename = os.path.join(
            self.save_dir,
            f"person_{track_id}.jpg",
        )

        cv2.imwrite(
            filename,
            crop,
        )

        info["saved"] = True

    def finalize(self):

        for (
            track_id,
            info,
        ) in list(
            self.tracks.items()
        ):

            self._save_crop(
                track_id,
                info,
            )