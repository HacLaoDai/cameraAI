"""
Công cụ quản lý người trong MongoDB collection `persons`
(dùng cho PersonDBRecognizer - cách 2: insightface + MongoDB).

VÍ DỤ:

    # Xem danh sách
    python manage_persons.py list

    # Thêm người mới từ ảnh có sẵn
    python3.10 manage_persons.py add --name "khanh" --type nhan_vien --sex 0 --age 21 \
        --images /home/lychien/Downloads/nhanvien.jpg

    # Thêm người mới từ ảnh bằng URL-
    python3.10 manage_persons.py add --name "testURL" --image-urls "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5nJNGcJ3uCTZM1JSbD-61H0kFREhtDYNtUgr0wrju6A&s=10"

    # Thêm người mới bằng webcam
    python3.10 manage_persons.py add --name "Chien" --webcam --num-captures 5

    # Sửa thông tin (không đổi ảnh/embedding)
    python manage_persons.py edit --name "Nhung" --new-name "Nguyen Thi Nhung" --age 23

    # Sửa + enroll lại embedding từ đầu (xóa embedding cũ, thêm ảnh mới)
    python manage_persons.py edit --name "Nhung" --clear-embeddings --webcam --num-captures 5

    # Thêm ảnh/embedding bổ sung (KHÔNG xóa embedding cũ)
    python manage_persons.py edit --name "Nhung" --images anh_moi.jpg
    python manage_persons.py edit --name "Nhung" --image-urls "https://example.com/anh_moi.jpg"

    # Xóa hẳn 1 người
    python3.10 manage_persons.py delete --name "chien"
    python manage_persons.py delete --person-id 1769051239 --yes
"""

import argparse

import cv2
import numpy as np
import requests

from detectors.detect_face import ArcFaceExtractor
from database import task_db


# ======================================================
# HELPERS - trích embedding từ ảnh / webcam / url
# ======================================================
def get_largest_embedding(extractor, img):
    """Lấy embedding của khuôn mặt LỚN NHẤT trong ảnh (thường rõ nét nhất)."""
    embeddings, bboxes = extractor.extract_embeddings(img)

    if len(embeddings) == 0:
        return None

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    idx = areas.index(max(areas))
    return embeddings[idx]


def download_image_from_url(url, timeout=10):
    """Tải ảnh từ URL và decode thành ảnh OpenCV (BGR). Trả về None nếu lỗi."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Không tải được ảnh từ URL: {url} ({e})")
        return None

    img_array = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[WARN] Dữ liệu tải về không phải ảnh hợp lệ: {url}")
        return None

    return img


def capture_embeddings_from_images(extractor, image_paths):
    embeddings = []

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"[WARN] Không đọc được ảnh: {path}")
            continue

        emb = get_largest_embedding(extractor, img)

        if emb is None:
            print(f"[WARN] Không tìm thấy khuôn mặt trong ảnh: {path}")
            continue

        embeddings.append(emb)
        print(f"[OK] Trích embedding từ: {path}")

    return embeddings


def capture_embeddings_from_urls(extractor, urls):
    embeddings = []

    for url in urls:
        img = download_image_from_url(url)

        if img is None:
            continue

        emb = get_largest_embedding(extractor, img)

        if emb is None:
            print(f"[WARN] Không tìm thấy khuôn mặt trong ảnh (URL): {url}")
            continue

        embeddings.append(emb)
        print(f"[OK] Trích embedding từ URL: {url}")

    return embeddings


def capture_embeddings_from_webcam(extractor, num_captures, camera_index, window_title="Capture"):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam")

    embeddings = []
    print(f"Nhấn SPACE để chụp ({num_captures} ảnh cần chụp), nhấn Q để dừng sớm.")

    try:
        while len(embeddings) < num_captures:
            ret, frame = cap.read()

            if not ret:
                continue

            display = frame.copy()
            cv2.putText(
                display,
                f"Da chup: {len(embeddings)}/{num_captures} - SPACE de chup, Q de thoat",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_title, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                emb = get_largest_embedding(extractor, frame)

                if emb is None:
                    print("[WARN] Không tìm thấy khuôn mặt, thử lại")
                    continue

                embeddings.append(emb)
                print(f"[OK] Đã chụp {len(embeddings)}/{num_captures}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return embeddings


def resolve_person(person_id=None, name=None):
    """Tìm 1 người theo person_id (ưu tiên) hoặc theo tên."""
    if person_id is not None:
        person = task_db.find_person_by_person_id(person_id)
        if not person:
            raise SystemExit(f"Không tìm thấy person_id={person_id}")
        return person

    if name:
        person = task_db.find_person_by_name(name)
        if not person:
            raise SystemExit(f"Không tìm thấy người tên '{name}'")
        return person

    raise SystemExit("Cần chỉ định --person-id hoặc --name")


def get_new_embeddings(args, window_title="Capture"):
    """Trả về list embedding mới nếu người dùng có yêu cầu --images/--image-urls/--webcam, ngược lại []."""
    if not args.images and not args.image_urls and not args.webcam:
        return []

    extractor = ArcFaceExtractor(ctx_id=args.ctx_id)

    embeddings = []

    if args.images:
        embeddings.extend(capture_embeddings_from_images(extractor, args.images))

    if args.image_urls:
        embeddings.extend(capture_embeddings_from_urls(extractor, args.image_urls))

# code dungf web cam

    if args.webcam:
        embeddings.extend(
            capture_embeddings_from_webcam(
                extractor, args.num_captures, args.camera_index, window_title
            )
        )

    return embeddings


# ======================================================
# COMMAND: add
# ======================================================
def cmd_add(args):
    info = {
        "name": args.name,
        "type": args.type,
        "sex": args.sex,
        "age": args.age,
        "phone": args.phone,
        "email": args.email,
        "nation": args.nation,
    }

    embeddings = get_new_embeddings(args, window_title=f"Add - {args.name}")

    if not embeddings:
        raise SystemExit("Cần --images <đường dẫn...>, --image-urls <url...> hoặc --webcam để có embedding.")

    existing = task_db.find_person_by_name(args.name)

    if existing:
        print(f"[WARN] '{args.name}' đã tồn tại (person_id={existing['person_id']}) "
              f"-> sẽ THÊM embedding vào người này thay vì tạo mới.")
        person_id = existing["person_id"]
        for emb in embeddings:
            task_db.add_embedding_to_person(person_id, emb)
    else:
        first_emb, rest = embeddings[0], embeddings[1:]
        person_id = task_db.insert_new_person(info, first_emb)
        for emb in rest:
            task_db.add_embedding_to_person(person_id, emb)

    print(f"[DONE] '{args.name}' (person_id={person_id}) - tổng {len(embeddings)} embedding vừa thêm")


# ======================================================
# COMMAND: edit
# ======================================================
def cmd_edit(args):
    person = resolve_person(args.person_id, args.name)
    person_id = person["person_id"]

    # --- Cập nhật thông tin (chỉ set field nào được truyền vào) ---
    update_data = {}
    if args.new_name:
        update_data["name"] = args.new_name
    if args.type:
        update_data["type"] = args.type
    if args.sex is not None:
        update_data["sex"] = args.sex
    if args.age is not None:
        update_data["age"] = args.age
    if args.phone is not None:
        update_data["phone"] = args.phone
    if args.email is not None:
        update_data["email"] = args.email
    if args.nation:
        update_data["nation"] = args.nation

    if update_data:
        task_db.update_person(person_id, update_data)
        print(f"[OK] Đã cập nhật thông tin: {update_data}")

    # --- Xóa embedding cũ nếu được yêu cầu ---
    if args.clear_embeddings:
        task_db.clear_person_embeddings(person_id)
        print("[OK] Đã xóa toàn bộ embedding cũ")

    # --- Thêm embedding mới (nếu có --images/--image-urls/--webcam) ---
    embeddings = get_new_embeddings(args, window_title=f"Edit - {person['name']}")

    for emb in embeddings:
        task_db.add_embedding_to_person(person_id, emb)

    if embeddings:
        print(f"[OK] Đã thêm {len(embeddings)} embedding mới")

    if not update_data and not args.clear_embeddings and not embeddings:
        print("[WARN] Không có gì để sửa - hãy truyền ít nhất 1 field, --clear-embeddings, "
              "hoặc --images/--image-urls/--webcam")

    print(f"[DONE] person_id={person_id}")


# ======================================================
# COMMAND: delete
# ======================================================
def cmd_delete(args):
    person = resolve_person(args.person_id, args.name)
    person_id = person["person_id"]
    name = person["name"]

    if not args.yes:
        confirm = input(f"Xóa hẳn '{name}' (person_id={person_id})? Không thể hoàn tác. Gõ 'yes' để xác nhận: ")
        if confirm.strip().lower() != "yes":
            print("Đã hủy.")
            return

    task_db.delete_person(person_id)
    print(f"[DONE] Đã xóa '{name}' (person_id={person_id})")


# ======================================================
# COMMAND: list
# ======================================================
def cmd_list(args):
    persons = task_db.get_all_persons()

    if not persons:
        print("(chưa có người nào trong DB)")
        return

    print(f"{'person_id':<15} {'name':<25} {'type':<12} {'age':<5} {'#embeddings'}")
    print("-" * 70)

    for p in persons:
        print(
            f"{p.get('person_id', ''):<15} "
            f"{p.get('name', ''):<25} "
            f"{p.get('type', ''):<12} "
            f"{p.get('age', ''):<5} "
            f"{len(p.get('embeddings', []))}"
        )


# ======================================================
# MAIN / ARGPARSE
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="Quản lý persons (MongoDB) - thêm/sửa/xóa")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    p_add = sub.add_parser("add", help="Thêm người mới")
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--type", default="nhan_vien")
    p_add.add_argument("--sex", type=int, default=0)
    p_add.add_argument("--age", type=int, default=0)
    p_add.add_argument("--phone", default="")
    p_add.add_argument("--email", default="")
    p_add.add_argument("--nation", default="VN")
    p_add.add_argument("--images", nargs="*")
    p_add.add_argument("--image-urls", nargs="*", dest="image_urls",
                        help="Danh sách URL ảnh để tải về và trích embedding")
    p_add.add_argument("--webcam", action="store_true")
    p_add.add_argument("--num-captures", type=int, default=5)
    p_add.add_argument("--camera-index", type=int, default=0)
    p_add.add_argument("--ctx-id", type=int, default=-1)
    p_add.set_defaults(func=cmd_add)

    # --- edit ---
    p_edit = sub.add_parser("edit", help="Sửa thông tin / thêm embedding / enroll lại")
    p_edit.add_argument("--person-id", type=int)
    p_edit.add_argument("--name")
    p_edit.add_argument("--new-name")
    p_edit.add_argument("--type")
    p_edit.add_argument("--sex", type=int)
    p_edit.add_argument("--age", type=int)
    p_edit.add_argument("--phone")
    p_edit.add_argument("--email")
    p_edit.add_argument("--nation")
    p_edit.add_argument("--clear-embeddings", action="store_true",
                         help="Xóa hết embedding cũ trước khi thêm mới (nếu có --images/--image-urls/--webcam)")
    p_edit.add_argument("--images", nargs="*")
    p_edit.add_argument("--image-urls", nargs="*", dest="image_urls",
                         help="Danh sách URL ảnh để tải về và trích embedding")
    p_edit.add_argument("--webcam", action="store_true")
    p_edit.add_argument("--num-captures", type=int, default=5)
    p_edit.add_argument("--camera-index", type=int, default=0)
    p_edit.add_argument("--ctx-id", type=int, default=-1)
    p_edit.set_defaults(func=cmd_edit)

    # --- delete ---
    p_delete = sub.add_parser("delete", help="Xóa hẳn 1 người")
    p_delete.add_argument("--person-id", type=int)
    p_delete.add_argument("--name")
    p_delete.add_argument("--yes", action="store_true", help="Bỏ qua bước xác nhận")
    p_delete.set_defaults(func=cmd_delete)

    # --- list ---
    p_list = sub.add_parser("list", help="Xem danh sách người trong DB")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()