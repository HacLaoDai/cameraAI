"""
Công cụ quản lý camera trong MongoDB collection `cameras`
(dùng cho multi_main.py - thay vì khai báo cứng trong CAMERAS ở code).

Sau khi thêm/sửa/xóa, chỉ cần KHỞI ĐỘNG LẠI multi_main.py để nhận cấu hình
mới (multi_main.py load danh sách camera từ DB 1 lần lúc start).

VÍ DỤ:

    # Xem danh sách camera
    python3.10 manage_cameras.py list

    # Thêm camera webcam mặc định của máy (cổng VÀO)
    python3.10 manage_cameras.py add --serial webcam-02 --channel webcam \
        --url 0 --in --address CS1

    # Ví dụ build chuỗi RTSP URL trong code (thay bằng giá trị thật, lấy từ
    # biến môi trường/file config, KHÔNG hardcode trong source):
    #   username = os.environ["CAM_USER"]
    #   password = os.environ["CAM_PASS"]
    #   ip = os.environ["CAM_IP"]
    #   rtsp_url = f"rtsp://{username}:{password}@{ip}:554/ch01/0"

    # Thêm camera RTSP (cổng RA)
    # LƯU Ý: KHÔNG dán user/pass thật vào lệnh này khi copy làm ví dụ/lưu
    # log/gửi cho người khác - RTSP URL chứa password ở dạng plain text.
    python manage_cameras.py add --serial cam-ra-01 --channel cam1 \
        --url "rtsp://<user>:<pass>@<camera_ip>:554/ch01/0" --in --address CS1

    # Thêm camera thuộc 1 recorder (NVR) đã tạo sẵn (theo mac recorder)
    python manage_cameras.py add --serial cam-vao-02 --channel cam2 \
        --url "rtsp://192.168.1.50:554/live2" --in --recorder-mac AA:BB:CC:DD:EE:FF

    # ---------------------------------------------------------------
    # CAMERA CẶP (paired) - 2 camera cùng quay 1 cửa, 1 IN + 1 OUT, dùng
    # để RESET dedup khi người thật sự đi qua cửa đó (xem multi_main.py -
    # RecognitionWorker._is_recently_logged()). NÊN đặt cặp 2 CHIỀU (mỗi
    # camera trỏ về camera kia) để reset hoạt động đúng bất kể ai bị dedup
    # trước:
    #     python manage_cameras.py edit --channel cam_cs1_in  --pair cam_cs1_out
    #     python3.10 manage_cameras.py edit --channel cam_cs1_out --pair cam_cs1_in
    #
    # Camera KHÔNG cặp với camera nào (vd camera giám sát chung, không gắn
    # với 1 cửa in/out cụ thể) thì bỏ qua --pair - mặc định paired_channel
    # = None, dedup chạy như bình thường (theo is_in, không có reset).
    #
    # Gỡ cặp: truyền chuỗi rỗng
    #     python manage_cameras.py edit --channel cam_cs1_in --pair ""
    # ---------------------------------------------------------------

    # Sửa url / đổi hướng / đổi tên channel
    python3.10 manage_cameras.py edit --serial cam_cs1_out_real --url "rtsp://admin:abcd1234%40%40@8aff08d3d4e5.sn.mynetname.net:554/Streaming/Channels/201"
    python manage_cameras.py edit --channel cam1 --out
    python manage_cameras.py edit --serial cam-ra-01 --new-channel cam1-ra

    # Tạm tắt / bật lại 1 camera (không cần xóa, multi_main.py sẽ bỏ qua
    # camera đang tắt khi load danh sách)
    python manage_cameras.py edit --channel cam1 --disable
    python manage_cameras.py edit --channel cam1 --enable

    # Xóa hẳn 1 camera
    python manage_cameras.py delete --channel cam1
    python manage_cameras.py delete --serial cam-ra-01 --yes
"""

import argparse

from database import task_db


# ======================================================
# HELPERS
# ======================================================
def resolve_camera(serial=None, channel=None):
    """Tìm 1 camera theo serial (ưu tiên) hoặc theo channel."""
    if serial:
        cam = task_db.get_camera_by_serial(serial)
        if not cam:
            raise SystemExit(f"Không tìm thấy camera serial='{serial}'")
        return cam

    if channel:
        cam = task_db.get_camera_by_channel(channel)
        if not cam:
            raise SystemExit(f"Không tìm thấy camera channel='{channel}'")
        return cam

    raise SystemExit("Cần chỉ định --serial hoặc --channel")


def resolve_recorder_id(recorder_mac):
    if not recorder_mac:
        return None

    recorder = task_db.get_recorder_by_mac(recorder_mac)

    if not recorder:
        raise SystemExit(f"Không tìm thấy recorder mac='{recorder_mac}' "
                          f"(tạo recorder trước, hoặc bỏ --recorder-mac)")

    return recorder["_id"]


def validate_pair_target(pair_channel, own_channel=None):
    """Nếu --pair được truyền (khác None) và khác rỗng, kiểm tra channel đó
    thật sự tồn tại trong DB - tránh cấu hình treo (trỏ tới channel không
    có thật) khiến logic reset dedup không bao giờ kích hoạt được mà không
    có cảnh báo gì."""
    if not pair_channel:
        return  # None hoặc "" (gỡ cặp) -> không cần kiểm tra

    if own_channel is not None and pair_channel == own_channel:
        raise SystemExit("--pair không được trỏ vào chính channel của camera đó.")

    target = task_db.get_camera_by_channel(pair_channel)
    if not target:
        raise SystemExit(
            f"--pair '{pair_channel}' không tồn tại trong DB - hãy thêm camera "
            f"đó trước, hoặc kiểm tra lại tên channel."
        )


# ======================================================
# COMMAND: add
# ======================================================
def cmd_add(args):
    if task_db.get_camera_by_serial(args.serial):
        raise SystemExit(f"Serial '{args.serial}' đã tồn tại.")

    if task_db.get_camera_by_channel(args.channel):
        raise SystemExit(
            f"Channel '{args.channel}' đã được dùng bởi camera khác - "
            f"channel phải là DUY NHẤT (được dùng để join với face_events)."
        )

    recorder_id = resolve_recorder_id(args.recorder_mac)

    validate_pair_target(args.pair, own_channel=args.channel)

    camera_id = task_db.create_camera(
        serial=args.serial,
        channel=args.channel,
        is_in=(args.direction == "in"),
        url=args.url,
        address=args.address,
        ip=args.ip,
        status=not args.disabled,
        recorder_id=recorder_id,
        # Camera cặp (cùng 1 cửa, hướng ngược lại) - dùng để reset dedup
        # khi người thật sự đi qua cửa đó, xem multi_main.py. None nếu
        # camera này không thuộc cặp nào (--pair không được truyền).
        paired_channel=args.pair or None,
    )

    direction_label = "VÀO" if args.direction == "in" else "RA"
    pair_label = f", cặp với '{args.pair}'" if args.pair else ""
    print(
        f"[DONE] Đã thêm camera '{args.channel}' "
        f"(serial={args.serial}, url={args.url}, cổng {direction_label}{pair_label}, "
        f"_id={camera_id})"
    )


# ======================================================
# COMMAND: edit
# ======================================================
def cmd_edit(args):
    camera = resolve_camera(args.serial, args.channel)
    camera_id = camera["_id"]

    update_data = {}

    if args.new_channel:
        if task_db.get_camera_by_channel(args.new_channel):
            raise SystemExit(f"Channel '{args.new_channel}' đã được dùng bởi camera khác.")
        update_data["channel"] = args.new_channel

    if args.url is not None:
        update_data["url"] = args.url

    if args.ip is not None:
        update_data["ip"] = args.ip

    if args.direction == "in":
        update_data["is_in"] = True
    elif args.direction == "out":
        update_data["is_in"] = False

    if args.enable:
        update_data["status"] = True
    elif args.disable:
        update_data["status"] = False

    if args.recorder_mac is not None:
        # cho phép truyền chuỗi rỗng "" để GỠ liên kết recorder
        update_data["recorder_id"] = resolve_recorder_id(args.recorder_mac) if args.recorder_mac else None

    if args.pair is not None:
        # cho phép truyền chuỗi rỗng "" để GỠ cặp (paired_channel = None)
        validate_pair_target(args.pair, own_channel=camera["channel"])
        update_data["paired_channel"] = args.pair or None

    if not update_data:
        raise SystemExit(
            "Không có gì để sửa - hãy truyền ít nhất 1 trong: --url, --ip, "
            "--new-channel, --in/--out, --enable/--disable, --recorder-mac, --pair"
        )

    task_db.update_camera(camera_id, update_data)

    print(f"[DONE] Đã cập nhật camera '{camera['channel']}': {update_data}")


# ======================================================
# COMMAND: delete
# ======================================================
def cmd_delete(args):
    camera = resolve_camera(args.serial, args.channel)
    channel = camera["channel"]

    if not args.yes:
        confirm = input(
            f"Xóa hẳn camera '{channel}' (serial={camera['serial']})? "
            f"Không thể hoàn tác. Gõ 'yes' để xác nhận: "
        )
        if confirm.strip().lower() != "yes":
            print("Đã hủy.")
            return

    task_db.delete_camera(camera["_id"])
    print(f"[DONE] Đã xóa camera '{channel}'")


# ======================================================
# COMMAND: list
# ======================================================
def cmd_list(args):
    cams = task_db.get_all_cameras()

    if not cams:
        print("(chưa có camera nào trong DB)")
        return

    print(
        f"{'channel':<15} {'serial':<15} {'url':<35} "
        f"{'direction':<9} {'status':<8} {'paired_channel':<15} {'recorder_id'}"
    )
    print("-" * 115)

    for c in cams:
        direction = "VAO" if c.get("is_in") else "RA"
        status = "ON" if c.get("status") else "OFF"

        print(
            f"{c.get('channel', ''):<15} "
            f"{c.get('serial', ''):<15} "
            f"{str(c.get('url', '')):<35} "
            f"{direction:<9} "
            f"{status:<8} "
            f"{c.get('paired_channel') or '-':<15} "
            f"{c.get('recorder_id') or ''}"
        )


# ======================================================
# MAIN / ARGPARSE
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Quản lý cameras (MongoDB) - thêm/sửa/xóa")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Thêm camera mới")
    p_add.add_argument("--serial", required=True, help="Mã định danh vật lý, phải DUY NHẤT")
    p_add.add_argument("--channel", required=True,
                        help="Tên dùng trong multi_main.py (CAMERAS[].name) và join với face_events, phải DUY NHẤT")
    p_add.add_argument("--url", required=True,
                        help="Chuỗi mở camera: RTSP URL, hoặc số 0/1/2.. cho webcam local")
    p_add.add_argument("--address", required=True,
                        help="Cơ sở mà camera được đặt")
    p_add.add_argument("--ip", default=None)
    p_add.add_argument("--recorder-mac", default=None, help="MAC của recorder/NVR (nếu camera thuộc 1 đầu ghi)")
    p_add.add_argument("--disabled", action="store_true", help="Thêm nhưng để status=OFF (multi_main.py sẽ bỏ qua)")
    p_add.add_argument("--pair", default=None,
                        help="Channel của camera CẶP với camera này (cùng 1 cửa, hướng ngược lại) - "
                             "dùng để reset dedup khi người thật sự đi qua cửa. Bỏ trống nếu không thuộc cặp nào.")

    direction = p_add.add_mutually_exclusive_group(required=True)
    direction.add_argument("--in", dest="direction", action="store_const", const="in", help="Camera đặt ở cổng VÀO")
    direction.add_argument("--out", dest="direction", action="store_const", const="out", help="Camera đặt ở cổng RA")
    p_add.set_defaults(func=cmd_add)

    # --- edit ---
    p_edit = sub.add_parser("edit", help="Sửa thông tin camera")
    p_edit.add_argument("--serial")
    p_edit.add_argument("--channel")
    p_edit.add_argument("--new-channel", default=None)
    p_edit.add_argument("--url", default=None)
    p_edit.add_argument("--ip", default=None)
    p_edit.add_argument("--recorder-mac", default=None,
                         help="Đổi recorder liên kết. Truyền chuỗi rỗng \"\" để gỡ liên kết.")
    p_edit.add_argument("--pair", default=None,
                         help="Đổi channel CẶP với camera này. Truyền chuỗi rỗng \"\" để gỡ cặp.")

    edit_direction = p_edit.add_mutually_exclusive_group()
    edit_direction.add_argument("--in", dest="direction", action="store_const", const="in")
    edit_direction.add_argument("--out", dest="direction", action="store_const", const="out")
    p_edit.set_defaults(direction=None)

    edit_status = p_edit.add_mutually_exclusive_group()
    edit_status.add_argument("--enable", action="store_true")
    edit_status.add_argument("--disable", action="store_true")
    p_edit.set_defaults(func=cmd_edit)

    # --- delete ---
    p_delete = sub.add_parser("delete", help="Xóa hẳn 1 camera")
    p_delete.add_argument("--serial")
    p_delete.add_argument("--channel")
    p_delete.add_argument("--yes", action="store_true", help="Bỏ qua bước xác nhận")
    p_delete.set_defaults(func=cmd_delete)

    # --- list ---
    p_list = sub.add_parser("list", help="Xem danh sách camera trong DB")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()