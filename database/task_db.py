import os
import base64
import math
from io import BytesIO
from datetime import datetime

import cv2
from pymongo import MongoClient
from bson import ObjectId
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt


# ======================================================
# CONNECT MONGODB
# ======================================================
# Khuyến nghị: chuyển URI sang biến môi trường thay vì hardcode trong code
# (VD: URI = os.environ["MONGO_URI"]) để tránh lộ mật khẩu khi commit code.
URI = os.environ.get(
    "MONGO_URI",
    "mongodb://baoan_dev:5769boan20s12rui@103.159.51.61/baoan_dev",
)
client = MongoClient(URI)
db = client.get_default_database()

# ======================================================
# COLLECTIONS
# ======================================================
users = db["users"]
face_events = db["face_events"]
recorders = db["recorders"]
cameras = db["cameras"]
persons = db["persons"]

# ======================================================
# INDEXES
# ======================================================
recorders.create_index("mac", unique=True)
cameras.create_index("serial", unique=True)
cameras.create_index("recorder_id")
cameras.create_index("channel")
face_events.create_index("time_in")
face_events.create_index("person_id")
face_events.create_index("channel")
persons.create_index("name")
persons.create_index("person_id", unique=True)


# ======================================================
# USERS
# ======================================================
def get_all_users():
    return list(users.find({}))


# ======================================================
# IMAGE HELPERS (dùng để log ảnh từ pipeline camera - numpy array)
# ======================================================
def encode_image_to_base64(frame, ext=".jpg"):
    """
    Chuyển ảnh OpenCV (numpy array, BGR) sang chuỗi base64
    để lưu vào field image2 trong face_events.
    """
    success, buffer = cv2.imencode(ext, frame)

    if not success:
        raise ValueError("Không encode được ảnh sang " + ext)

    return base64.b64encode(buffer).decode("utf-8")


# ======================================================
# FACE EVENTS
# ======================================================
def save_face_to_db(face: dict):
    """
    face cần có:
        face_recognition: {"result": "matched"/khác, "person_id":, "name":, "type":}
        StartTime: unix timestamp (giây)
        Image2: chuỗi base64 của ảnh
        channel: tên/mã camera đã phát hiện ra người này (BẮT BUỘC để tính năng
                 vào/ra hoạt động, vì get_current_staying()/get_person_movements()
                 join theo field này với collection cameras)
    """
    rec = face.get("face_recognition", {})

    if rec.get("result") == "matched":
        person_id = rec["person_id"]
        name = rec["name"]
        group = rec["type"]
    else:
        person_id = -1
        name = "Người lạ"
        group = "nguoi_la"

    doc = {
        "person_id": person_id,
        "name": name,
        "check": False,
        "type": group,
        "channel": face.get("channel"),
        "time_in": datetime.fromtimestamp(face.get("StartTime") - 3600 * 7),
        "image2": face.get("Image2"),
        "created_at": datetime.utcnow(),
    }

    return face_events.insert_one(doc).inserted_id


def get_face_event_by_id(event_id: str):
    return face_events.find_one({"_id": ObjectId(event_id)})


def get_face_events(limit=100):
    return list(
        face_events.find({})
        .sort("time_in", -1)
        .limit(limit)
    )


def get_all_face_events_simple(limit=100):
    events = face_events.find(
        {},
        {
            "_id": 0,
            "person_id": 1,
            "name": 1,
            "time_in": 1,
            "time_out": 1,
        },
    ).sort("time_in", -1).limit(limit)

    result = []
    for e in events:
        name = e.get("name") if e.get("person_id") != -1 else "Người lạ"

        result.append({
            "name": name,
            "time_in": e.get("time_in"),
            "time_out": e.get("time_out"),
        })

    return result


def get_person_movements(person_id):
    pipeline = [
        {"$match": {"person_id": person_id}},
        {
            "$lookup": {
                "from": "cameras",
                "localField": "channel",
                "foreignField": "channel",
                "as": "cam",
            }
        },
        {"$unwind": "$cam"},
        {"$sort": {"time_in": -1}},
        {
            "$project": {
                "_id": 0,
                "time": "$time_in",
                "channel": "$channel",
                "is_in": "$cam.is_in",
            }
        },
    ]
    return list(face_events.aggregate(pipeline))


def get_face_events_filter(channel=None, person_type=None, skip=0, limit=20):
    query = {}

    if channel and channel != "all":
        query["channel"] = channel

    if person_type == "unknown":
        query["person_id"] = -1

    if person_type == "known":
        query["person_id"] = {"$ne": -1}

    total = face_events.count_documents(query)

    data = list(
        face_events.find(query, {
            "_id": 1,
            "name": 1,
            "time_in": 1,
            "image2": 1,
            "person_id": 1,
            "channel": 1,
        })
        .sort("time_in", -1)
        .skip(skip)
        .limit(limit)
    )

    return total, data


def get_current_staying():
    """
    Trả về danh sách người hiện đang "ở trong": lấy event mới nhất của mỗi
    person_id, chỉ giữ lại những ai event cuối cùng đó thuộc camera có is_in=True.
    """
    pipeline = [
        {
            "$lookup": {
                "from": "cameras",
                "localField": "channel",
                "foreignField": "channel",
                "as": "cam",
            }
        },
        {"$unwind": "$cam"},
        {"$sort": {"time_in": -1}},
        {
            "$group": {
                "_id": "$person_id",
                "last": {"$first": "$$ROOT"},
            }
        },
        {"$replaceRoot": {"newRoot": "$last"}},
        {"$match": {"cam.is_in": True}},
    ]

    return list(face_events.aggregate(pipeline))


def get_face_events_simple(limit=50):
    return list(
        face_events.find(
            {},
            {
                "_id": 1,
                "person_id": 1,
                "name": 1,
                "time_in": 1,
                "time_out": 1,
                "image2": 1,
            },
        )
        .sort("time_in", -1)
        .limit(limit)
    )


def get_face_events_by_person(person_id, limit=50):
    return list(
        face_events.find({"person_id": person_id})
        .sort("time_in", -1)
        .limit(limit)
    )


def get_unknown_faces(limit=50):
    return list(
        face_events.find({"person_id": -1})
        .sort("time_in", -1)
        .limit(limit)
    )


def update_face_event(event_id: str, data: dict):
    return face_events.update_one(
        {"_id": ObjectId(event_id)},
        {"$set": data},
    )


def delete_face_event(event_id: str):
    return face_events.delete_one({"_id": ObjectId(event_id)})


def delete_face_events_by_person(person_id):
    return face_events.delete_many({"person_id": person_id})


# ======================================================
# RECORDERS
# ======================================================
def create_recorder(
    mac,
    ip_public=None,
    ip_lan=None,
    address=None,
    account=None,
    password=None,
    port=None,
):
    doc = {
        "mac": mac,
        "ip_public": ip_public,
        "ip_lan": ip_lan,
        "address": address,
        "account": account,
        "password": password,
        "port": port,
        "created_at": datetime.utcnow(),
    }
    result = recorders.insert_one(doc)
    return result.inserted_id


def get_recorder_by_id(recorder_id):
    return recorders.find_one({"_id": ObjectId(recorder_id)})


def get_recorder_by_mac(mac: str):
    return recorders.find_one({"mac": mac})


def get_all_recorders():
    return list(recorders.find({}))


def update_recorder(recorder_id, data: dict):
    return recorders.update_one(
        {"_id": ObjectId(recorder_id)},
        {"$set": data},
    )


def delete_recorder(recorder_id):
    cameras.delete_many({"recorder_id": ObjectId(recorder_id)})
    return recorders.delete_one({"_id": ObjectId(recorder_id)})


# ======================================================
# CAMERAS
# ======================================================
def create_camera(
    serial,
    channel,
    is_in,
    url=None,
    ip=None,
    status=True,
    recorder_id=None,
):
    """
    is_in=True  -> camera đặt ở cổng VÀO
    is_in=False -> camera đặt ở cổng RA
    channel     -> mã định danh camera, PHẢI khớp với giá trị "channel"
                   dùng khi gọi save_face_to_db() từ pipeline camera.
    url         -> chuỗi để mở camera (vd "rtsp://.../live1", hoặc "0" cho
                   webcam mặc định). BẮT BUỘC để multi_main.py có thể tự
                   load danh sách camera từ DB (xem load_cameras_from_db
                   trong multi_main.py và manage_cameras.py để thêm/sửa/xóa).
    status      -> True = đang bật/đang dùng, False = tạm tắt (multi_main.py
                   chỉ chạy pipeline cho các camera có status=True).
    """
    doc = {
        "serial": serial,
        "channel": channel,
        "is_in": is_in,
        "url": url,
        "ip": ip,
        "status": status,
        "recorder_id": ObjectId(recorder_id) if recorder_id else None,
        "created_at": datetime.utcnow(),
    }
    result = cameras.insert_one(doc)
    return result.inserted_id


def get_camera_by_id(camera_id):
    return cameras.find_one({"_id": ObjectId(camera_id)})


def get_camera_by_serial(serial: str):
    return cameras.find_one({"serial": serial})


def get_camera_by_channel(channel: str):
    return cameras.find_one({"channel": channel})


def get_cameras_by_recorder(recorder_id):
    return list(cameras.find({"recorder_id": ObjectId(recorder_id)}))


def get_all_cameras():
    return list(cameras.find({}))


def get_active_cameras():
    """Chỉ lấy camera đang bật (status=True) - dùng để load vào multi_main.py."""
    return list(cameras.find({"status": True}))


def update_camera(camera_id, data: dict):
    return cameras.update_one(
        {"_id": ObjectId(camera_id)},
        {"$set": data},
    )


def delete_camera(camera_id):
    return cameras.delete_one({"_id": ObjectId(camera_id)})


def get_cameras_with_recorder():
    pipeline = [
        {
            "$lookup": {
                "from": "recorders",
                "localField": "recorder_id",
                "foreignField": "_id",
                "as": "recorder",
            }
        },
        {"$unwind": "$recorder"},
    ]
    return list(cameras.aggregate(pipeline))


def deletl_all(db_):
    db_.cameras.delete_many({})
    db_.recorders.delete_many({})


# ======================================================
# SHOW (chỉ dùng khi debug thủ công, không gọi khi import module)
# ======================================================
def base64_to_pil(base64_str):
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def draw_info_on_image(img, name, time_in):
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    text = f"{name} | {time_in.strftime('%Y-%m-%d %H:%M:%S')}"

    draw.rectangle((0, 0, img.width, 40), fill=(0, 0, 0))
    draw.text((10, 8), text, fill=(255, 255, 255), font=font)

    return img


def show_face_images(events, cols=4):
    rows = math.ceil(len(events) / cols)
    plt.figure(figsize=(cols * 4, rows * 4))

    for i, ev in enumerate(events):
        img = base64_to_pil(ev["image2"])
        img = draw_info_on_image(
            img,
            ev.get("name", "Unknown"),
            ev["time_in"],
        )

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================
# PERSONS (Face Identity DB)
# ======================================================
def find_person_by_name(name):
    return persons.find_one({"name": name})


def find_person_by_person_id(person_id):
    return persons.find_one({"person_id": int(person_id)})


def insert_new_person(info, embedding):
    doc = {
        "person_id": int(datetime.utcnow().timestamp()),
        "name": info["name"],
        "type": info["type"],
        "sex": info["sex"],
        "age": info["age"],
        "phone": info["phone"],
        "email": info["email"],
        "nation": info["nation"],
        "embeddings": [embedding.tolist()],
        "created_at": datetime.utcnow(),
    }
    persons.insert_one(doc)
    return doc["person_id"]


def add_embedding_to_person(person_id, embedding):
    persons.update_one(
        {"person_id": int(person_id)},
        {"$push": {"embeddings": embedding.tolist()}},
    )


def update_person(person_id, data: dict):
    """Sửa thông tin (name, type, sex, age, phone, email, nation...)."""
    return persons.update_one(
        {"person_id": int(person_id)},
        {"$set": data},
    )


def clear_person_embeddings(person_id):
    """Xóa hết embedding cũ của 1 người (dùng khi muốn enroll lại từ đầu)."""
    return persons.update_one(
        {"person_id": int(person_id)},
        {"$set": {"embeddings": []}},
    )


def delete_person(person_id):
    """Xóa hẳn 1 người khỏi persons."""
    return persons.delete_one({"person_id": int(person_id)})


def get_all_persons():
    return list(persons.find({}))


def search_person_by_name(prefix, limit=10):
    return list(persons.find(
        {"name": {"$regex": f"^{prefix}", "$options": "i"}},
        {"_id": 0, "person_id": 1, "name": 1},
    ).limit(limit))


# ======================================================
# CHỈ chạy khi gọi trực tiếp file này (python task_db.py)
# Import module này ở nơi khác (vd: main_multi.py) sẽ KHÔNG tự động
# mở popup matplotlib -> an toàn khi ghép vào pipeline.
# ======================================================
if __name__ == "__main__":
    events = get_face_events_simple(limit=20)
    # show_face_images(events)
    print(get_all_persons())