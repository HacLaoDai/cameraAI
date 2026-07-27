"""
Configure Occupancy Camera Zone

Usage:
    python configure_zone_camera.py --channel cam_zone_1

Controls:
    Left Click   : Add point
    Right Click  : Undo last point
    TAB          : Switch zone_out <-> zone_in
    C            : Clear current zone
    R            : Reset all zones
    S            : Save to MongoDB
    Q            : Quit without save
"""

import argparse
import cv2
import numpy as np

from database import task_db

current_zone = "zone_out"

zones = {
    "zone_out": [],
    "zone_in": []
}


def mouse_callback(event, x, y, flags, param):
    global zones

    if event == cv2.EVENT_LBUTTONDOWN:
        zones[current_zone].append((x, y))
        print(f"[{current_zone}] Add point ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if zones[current_zone]:
            removed = zones[current_zone].pop()
            print(f"[{current_zone}] Remove point {removed}")


def draw_zone(frame, overlay, name, pts, color):

    if len(pts) >= 3:
        cv2.fillPoly(
            overlay,
            [np.array(pts, np.int32)],
            color
        )

    for idx, p in enumerate(pts):

        cv2.circle(frame, p, 6, color, -1)

        cv2.putText(
            frame,
            str(idx + 1),
            (p[0] + 8, p[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    if len(pts) > 1:
        cv2.polylines(
            frame,
            [np.array(pts, np.int32)],
            len(pts) >= 3,
            color,
            3 if name == current_zone else 2
        )

    if pts:
        x, y = pts[0]

        cv2.putText(
            frame,
            f"{name}",
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


def draw_ui(frame):

    overlay = frame.copy()

    draw_zone(
        frame,
        overlay,
        "zone_out",
        zones["zone_out"],
        (255, 0, 255)
    )

    draw_zone(
        frame,
        overlay,
        "zone_in",
        zones["zone_in"],
        (0, 255, 255)
    )

    cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0,
        frame
    )

    cv2.rectangle(frame, (10, 10), (900, 150), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Current Zone : {current_zone}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"ZONE_OUT Points : {len(zones['zone_out'])}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"ZONE_IN Points : {len(zones['zone_in'])}",
        (320, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Left Click:Add  Right Click:Undo  TAB:Switch",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "C:Clear Current  R:Reset All  S:Save  Q:Quit",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return frame


def main():

    global current_zone
    global zones

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channel",
        required=True,
        help="Camera channel in MongoDB"
    )

    args = parser.parse_args()

    cam = task_db.get_camera_by_channel(args.channel)

    if not cam:
        print(
            f"Camera channel='{args.channel}' not found."
        )
        return

    url = cam.get("url")

    if isinstance(url, str):
        stripped = url.strip()

        if stripped.lstrip("-").isdigit():
            url = int(stripped)

    if (
        isinstance(url, str)
        and url.lower().startswith("rtsp")
    ):
        cap = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG
        )
    else:
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("Cannot open camera:", url)
        return

    cv2.namedWindow("Configure Zone")
    cv2.setMouseCallback(
        "Configure Zone",
        mouse_callback
    )

    print("=" * 60)
    print("CONFIGURE OCCUPANCY CAMERA")
    print("Camera:", args.channel)
    print("=" * 60)
    print("Start with zone_out")
    print("Person path:")
    print("zone_out  --->  zone_in  = ENTRY")
    print("zone_in   --->  zone_out = EXIT")
    print("=" * 60)

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Camera disconnected.")
                break

            frame = draw_ui(frame)

            cv2.imshow(
                "Configure Zone",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == 9:

                current_zone = (
                    "zone_in"
                    if current_zone == "zone_out"
                    else "zone_out"
                )

                print(
                    f"Current zone -> {current_zone}"
                )

            elif key == ord("c"):

                zones[current_zone].clear()

                print(
                    f"Clear {current_zone}"
                )

            elif key == ord("r"):

                zones = {
                    "zone_out": [],
                    "zone_in": []
                }

                print("Reset all zones")

            elif key == ord("s"):

                if len(zones["zone_out"]) < 3:
                    print(
                        "zone_out requires at least 3 points"
                    )
                    continue

                if len(zones["zone_in"]) < 3:
                    print(
                        "zone_in requires at least 3 points"
                    )
                    continue

                task_db.configure_occupancy_camera(
                    args.channel,
                    zones["zone_in"],
                    zones["zone_out"]
                )

                print()
                print("================================")
                print("SAVE SUCCESS")
                print("Channel :", args.channel)
                print(
                    f"zone_out : {len(zones['zone_out'])} points"
                )
                print(
                    f"zone_in  : {len(zones['zone_in'])} points"
                )
                print("================================")

                break

            elif key == ord("q"):

                print("Quit without saving.")
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()