import os
import json
import base64
import math
import glob
import time
import logging
from io import BytesIO
from datetime import datetime, timedelta

import cv2
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from PIL import ImageDraw, ImageFont, Image
# matplotlib CHỈ dùng trong show_face_images() (xem debug thủ công phía dưới
# file) - import ngay tại đó thay vì ở đây, để multi_main.py (chạy 24/7,
# không bao giờ gọi show_face_images) không phải tải matplotlib + backend
# GUI của nó (có thể tự chọn Qt) vào RAM một cách không cần thiết.
from bson import ObjectId

# ======================================================
# LOGGING
# ======================================================
logger = logging.getLogger("task_db")

# ======================================================
# CONNECT MONGODB
# ======================================================
# QUAN TRỌNG: KHÔNG hardcode user/pass ở đây. URI BẮT BUỘC phải được truyền
# qua biến môi trường MONGO_URI (vd trong file .env / systemd EnvironmentFile),
# không commit vào source control.
#
# Cách chạy:
#   export MONGO_URI="mongodb://<user>:<pass>@<host>/<db>"
#   python multi_main.py
#
# Nếu mật khẩu Mongo trước đây từng bị hardcode trong file này và đã từng
# commit/chia sẻ ra ngoài (kể cả gửi cho ai đó ngoài đội dev) -> coi như đã
# lộ, cần ĐỔI MẬT KHẨU MONGO NGAY, không chỉ xoá dòng code này là đủ.
URI = os.environ.get(
    "MONGO_URI",
    "mongodb://baoan_dev:5769boan20s12rui@103.159.51.61/baoan_dev",
)

client = MongoClient(URI)
db = client.get_default_database()

# ======================================================
# PENDING-WRITE BUFFER (retry khi mất kết nối Mongo)
# ======================================================
# Nếu ghi face_events thất bại (vd mất mạng WAN tới Mongo), thay vì MẤT
# LUÔN event đó, ta lưu tạm ra file JSON trong PENDING_DIR. Gọi
# flush_pending_face_events() định kỳ (vd mỗi vài phút trong multi_main.py)
# để thử ghi lại các event còn tồn đọng.
PENDING_DIR = os.environ.get("PENDING_EVENTS_DIR", "pending_face_events")
os.makedirs(PENDING_DIR, exist_ok=True)


def _buffer_face_event_to_disk(doc: dict):
    """Lưu tạm 1 face_event ra đĩa khi ghi Mongo thất bại, để retry sau."""
    safe_doc = dict(doc)
    # datetime không tự serialize JSON được -> chuyển sang isoformat
    for k, v in safe_doc.items():
        if isinstance(v, datetime):
            safe_doc[k] = v.isoformat()

    filename = os.path.join(
        PENDING_DIR, f"{time.time()}_{os.getpid()}.json"
    )
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(safe_doc, f)
        logger.warning("Đã buffer 1 face_event ra %s (Mongo tạm thời lỗi)", filename)
    except Exception as e:
        # Nếu cả ghi file cũng lỗi thì đành chịu mất - nhưng ít nhất log rõ
        logger.error("Không buffer được face_event ra đĩa: %s", e)


def flush_pending_face_events():
    """
    Thử ghi lại các face_event đang tồn đọng trong PENDING_DIR (do lần trước
    ghi Mongo thất bại). Gọi định kỳ từ multi_main.py (vd mỗi 60-120s).
    Trả về số event đã ghi thành công.
    """
    files = sorted(glob.glob(os.path.join(PENDING_DIR, "*.json")))
    if not files:
        return 0

    success = 0
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            if isinstance(doc.get("time_in"), str):
                doc["time_in"] = datetime.fromisoformat(doc["time_in"])
            if isinstance(doc.get("created_at"), str):
                doc["created_at"] = datetime.fromisoformat(doc["created_at"])

            face_events.insert_one(doc)
            os.remove(path)
            success += 1
        except PyMongoError as e:
            # Mongo vẫn đang lỗi -> giữ nguyên file, thử lại lần sau
            logger.warning("Mongo vẫn chưa ghi được, giữ lại %s: %s", path, e)
            break
        except Exception as e:
            # File hỏng/parse lỗi -> log và bỏ qua file này (không lặp vô hạn)
            logger.error("Bỏ qua file pending lỗi %s: %s", path, e)

    if success:
        logger.info("Đã flush %d face_event tồn đọng lên Mongo", success)

    return success

# ======================================================
# COLLECTIONS
# ======================================================
users = db["users"]
face_events = db["face_events"]
recorders = db["recorders"]
cameras = db["cameras"]
persons = db["persons"]
person_sessions = db["person_sessions"]

# zone_events / zone_occupancy: dùng RIÊNG cho camera "occupancy" (thấy cả
# 2 chiều, đếm ẨN DANH bằng ZoneCounter - xem zone_counter.py) - KHÔNG liên
# quan gì tới face_events/persons/person_sessions ở trên (những cái đó chỉ
# dùng cho camera "identity" cố định 1 chiều, cần nhận diện mặt).
zone_events = db["zone_events"]
zone_occupancy = db["zone_occupancy"]

# ======================================================
# INDEXES
# ======================================================
recorders.create_index("mac", unique=True)
cameras.create_index("serial", unique=True)
cameras.create_index("recorder_id")
cameras.create_index("channel")
face_events.create_index("time_in")
face_events.create_index("person_id")
face_events.create_index("identity")
face_events.create_index("channel")
persons.create_index("name")
persons.create_index("person_id", unique=True)
# person_sessions: truy vấn chính là "tìm session ĐANG MỞ của 1 người"
# ((person_id, status) - compound index) và thống kê theo entry_time.
person_sessions.create_index([("person_id", 1), ("status", 1)])
person_sessions.create_index([("person_id", 1), ("updated_at", -1)])
person_sessions.create_index("entry_time")
person_sessions.create_index("address")

zone_events.create_index("channel")
zone_events.create_index("time")
zone_occupancy.create_index("address", unique=True)


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
        identity (tuỳ chọn): danh tính ỔN ĐỊNH của người này trong lượt này -
                 person_id thật (int) nếu đã enroll, HOẶC temp_id (str, vd
                 "unk_abc123") do UnknownGallery cấp nếu là người lạ, HOẶC
                 None nếu không xác định được (không detect ra mặt nào).

                 QUAN TRỌNG: trước đây MỌI người lạ đều lưu person_id=-1
                 giống hệt nhau -> get_current_staying() group theo
                 person_id sẽ gộp NHIỀU người lạ khác nhau thành 1 dòng
                 duy nhất, đếm thiếu số lượng thực tế đang ở trong homestay.
                 Field "identity" này tách riêng từng người lạ ra để đếm
                 đúng, trong khi "person_id" vẫn giữ nguyên -1 cho họ (không
                 đổi ý nghĩa cũ, không ảnh hưởng các hàm lọc known/unknown
                 hiện có dựa theo person_id).
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

    identity = face.get("identity")
    if identity is None and rec.get("result") == "matched":
        # Người đã enroll: identity luôn = person_id thật, không phụ thuộc
        # caller có truyền identity vào hay không (an toàn cho code gọi cũ).
        identity = person_id

    channel = face.get("channel")

    camera = get_camera_by_channel(channel)

    address = None
    if camera:
        address = camera.get("address")

    doc = {
        "person_id": person_id,
        "identity": identity,
        "name": name,
        "check": False,
        "type": group,

        "channel": channel,
        "address": address,

        "time_in": datetime.now(),
        "image2": face.get("Image2"),
        # "created_at": datetime.utcnow(),
    }

    try:
        return face_events.insert_one(doc).inserted_id
    except PyMongoError as e:
        # Mất kết nối Mongo (mạng WAN chập chờn...) -> KHÔNG ném lỗi làm
        # crash pipeline, cũng KHÔNG âm thầm mất event. Lưu tạm ra đĩa để
        # flush_pending_face_events() ghi bù lại sau khi Mongo hồi phục.
        logger.error("Ghi face_event lên Mongo thất bại (%s) -> buffer ra đĩa", e)
        _buffer_face_event_to_disk(doc)
        return None


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


def serialize_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, list):
        return [serialize_objectid(x) for x in obj]

    if isinstance(obj, dict):
        return {k: serialize_objectid(v) for k, v in obj.items()}

    return obj

def get_current_staying(max_stay_hours=None):
    """
    Trả về danh sách người hiện đang "lưu trú" tại TỪNG CƠ SỞ (address).

    Logic: với mỗi cặp (identity, address) - tức 1 NGƯỜI CỤ THỂ tại 1 cơ sở
    cụ thể - lấy event GẦN NHẤT của họ tại chính cơ sở đó. Nếu event gần
    nhất này là is_in=True (vào) -> coi là đang lưu trú TẠI CƠ SỞ ĐÓ.

    QUAN TRỌNG - group theo "identity" (KHÔNG phải "person_id" như bản
    trước): person_id chỉ phân biệt được người ĐÃ ENROLL (mỗi người 1
    person_id thật). Mọi khách LẠ/chưa enroll đều lưu person_id=-1 GIỐNG
    HỆT NHAU -> nếu group theo person_id, 5 khách lạ đang ở cùng lúc sẽ bị
    gộp thành 1 dòng duy nhất, đếm THIẾU số lượng thực tế rất nhiều - sai
    ngay mục đích chính là đếm số người trong homestay.

    "identity" (xem save_face_to_db()) tách riêng: person_id thật cho
    người đã enroll, temp_id ổn định (từ UnknownGallery) cho từng khách lạ
    khác nhau -> mỗi người, dù lạ hay quen, đều có 1 khoá đếm riêng.

    Bỏ qua các event không có identity (identity=None - trường hợp hiếm,
    xảy ra khi track được lưu nhưng không detect ra khuôn mặt nào) vì
    không có khoá ổn định để tính là "cùng 1 người" hay "người mới" một
    cách đáng tin cậy.

    max_stay_hours (tuỳ chọn, khuyến nghị dùng): nếu set, sẽ LOẠI BỎ khỏi
    kết quả những lượt "lưu trú" đã kéo dài quá N giờ. Lý do cần cái này:
    nếu camera cổng ra bị lỗi/mất kết nối hoặc người đó rời qua cửa không
    có camera, hệ thống sẽ mãi mãi coi họ là "đang lưu trú" dù thực tế đã
    rời đi từ lâu (dữ liệu "ma" tích tụ theo thời gian, làm số đếm ngày
    càng lệch xa thực tế). Ví dụ set = 24 để tự động ẩn các lượt vào quá
    24h không có ra tương ứng - NÊN đặt giá trị này vì mục đích đếm số
    lượng nhạy với dữ liệu "ma" hơn nhiều so với mục đích chấm công/log.
    """
    pipeline = [
        {"$match": {"identity": {"$ne": None}}},
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
                "_id": {"identity": "$identity", "address": "$cam.address"},
                "last": {"$first": "$$ROOT"},
            }
        },
        {"$replaceRoot": {"newRoot": "$last"}},
        {"$match": {"cam.is_in": True}},
    ]

    if max_stay_hours is not None:
        cutoff = datetime.utcnow() - timedelta(hours=max_stay_hours)
        pipeline.append({"$match": {"time_in": {"$gte": cutoff}}})

    events = list(face_events.aggregate(pipeline))

    return serialize_objectid(events)


def get_current_staying_count(address=None, max_stay_hours=24):
    """
    Tiện ích đơn giản: trả về SỐ LƯỢNG người hiện đang lưu trú (int) - dùng
    thẳng cho mục đích "đếm số lượng trong homestay" thay vì phải tự đếm
    len(...) mỗi nơi gọi.

    address=None -> đếm TẤT CẢ cơ sở gộp lại. Truyền address cụ thể để chỉ
    đếm riêng 1 cơ sở (homestay).

    max_stay_hours mặc định = 24 (khác với get_current_staying() mặc định
    None) vì hàm này dùng cho hiển thị số liệu TRỰC TIẾP cho người dùng -
    nên chủ động lọc "ma" thay vì để None và tự phải nhớ truyền tay.
    """
    staying = get_current_staying(max_stay_hours=max_stay_hours)
    if address is not None:
        staying = [s for s in staying if s.get("cam", {}).get("address") == address]
    return len(staying)


def get_current_staying_by_address(address, max_stay_hours=None):
    """
    Giống get_current_staying(), nhưng chỉ lọc riêng cho 1 cơ sở cụ thể -
    tiện dùng khi hiển thị màn hình "ai đang ở cơ sở X" thay vì tất cả.
    """
    all_staying = get_current_staying(max_stay_hours=max_stay_hours)
    return [s for s in all_staying if s.get("cam", {}).get("address") == address]


def get_recent_dedup_state(cooldown_seconds):
    """
    Trả về {(identity, is_in): last_StartTime_unix} cho các face_events có
    identity xác định (đã enroll HOẶC người lạ có temp_id ổn định) còn nằm
    trong vòng cooldown_seconds gần đây.

    Dùng để RecognitionWorker (multi_main.py) PRELOAD lại
    `last_logged_by_person` mỗi khi chương trình khởi động/khởi động lại -
    nếu không, dict dedup trong RAM bị reset về rỗng sau mỗi lần restart,
    khiến người vừa được ghi DB trước khi tắt chương trình bị ghi trùng lại
    ngay khi chạy lại, dù chưa hết cooldown.

    ĐỔI từ "person_id" sang "identity": RecognitionWorker._finalize_event()
    dùng dedup_key = (identity, is_in) - identity có thể là person_id thật
    HOẶC temp_id (str) của người lạ. Bản cũ lọc theo person_id != -1 nên
    CHỈ preload được người đã enroll, bỏ sót hoàn toàn người lạ -> sau mỗi
    lần restart, người lạ vừa rời đi trước đó vài giây có thể bị ghi trùng
    ngay lập tức dù chưa hết cooldown.

    LƯU Ý VỀ THỜI GIAN: time_in được lưu bởi save_face_to_db() theo công thức
        time_in = datetime.fromtimestamp(StartTime - 3600*7)
    tức là đã áp dụng 1 phép biến đổi lệch múi giờ. Để so sánh đúng với
    time.time() dùng trong RecognitionWorker._is_recently_logged(), ta phải
    NGHỊCH ĐẢO đúng công thức đó (giả định server không đổi timezone giữa
    các lần chạy - đúng với 1 server cố định):
        StartTime = time_in.timestamp() + 3600*7
    """
    if not cooldown_seconds:
        return {}

    # cutoff tính theo ĐÚNG công thức lệch giờ mà save_face_to_db() dùng, để
    # so sánh nhất quán với dữ liệu đã lưu trong time_in
    cutoff = datetime.fromtimestamp(time.time() - cooldown_seconds - 3600 * 7)

    pipeline = [
        {"$match": {"identity": {"$ne": None}, "time_in": {"$gte": cutoff}}},
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
                "_id": {"identity": "$identity", "is_in": "$cam.is_in"},
                "last_time_in": {"$first": "$time_in"},
            }
        },
    ]

    result = {}
    try:
        for doc in face_events.aggregate(pipeline):
            key = (doc["_id"]["identity"], doc["_id"]["is_in"])
            result[key] = doc["last_time_in"].timestamp() + 3600 * 7
    except PyMongoError as e:
        # Nếu Mongo lỗi ngay lúc khởi động thì bỏ qua preload (dedup sẽ coi
        # như chưa ai được ghi - chấp nhận được, còn hơn crash lúc start)
        logger.error("Không preload được dedup state từ Mongo: %s", e)

    return result


def get_face_events_simple(limit=50):
    return list(
        face_events.find(
            {},
            {
                "_id": 1,
                "person_id": 1,
                "name": 1,
                "check":1,
                "type":1,
                "channel":1,
                "address":1,
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
# PERSON SESSIONS (đang ở trong / lịch sử vào-ra theo camera IN/OUT)
# ======================================================
# Đây là 1 collection RIÊNG, TÁCH BIỆT hoàn toàn với face_events:
#   - face_events: LỊCH SỬ THÔ, ghi lại MỌI LẦN thấy mặt (ảnh, camera nào,
#     lúc nào) - không có khái niệm "phiên" (session), không dùng để đếm
#     "đang ở trong" trực tiếp.
#   - person_sessions: TRẠNG THÁI, mỗi document là 1 LƯỢT VÀO-RA hoàn
#     chỉnh của 1 người tại 1 cơ sở (address). Dùng để đếm "đang có bao
#     nhiêu người trong khu vực" và "hôm nay có bao nhiêu lượt vào" cực
#     nhanh (chỉ count_documents/distinct, không cần aggregate như
#     get_current_staying() dựa trên face_events).
#
# "person_id" trong collection này thực chất là IDENTITY của hệ thống
# (xem save_face_to_db()) - có thể là person_id thật (int, đã enroll) HOẶC
# temp_id (str, người lạ do UnknownGallery cấp). Đặt tên field là
# "person_id" cho khớp đúng yêu cầu/pseudocode, nhưng về bản chất nhận mọi
# kiểu identity.
#
# entry_time/exit_time lưu dạng datetime (KHÔNG lưu chuỗi "09:31") vì còn
# cần so sánh khoảng ngày ($gte start_day, $lt end_day) cho các hàm thống
# kê bên dưới - chuỗi giờ:phút không so sánh khoảng ngày được. Muốn hiển
# thị dạng "09:31" cho người dùng thì format ở tầng hiển thị
# (vd entry_time.strftime("%H:%M")), không lưu sẵn dạng đó trong DB.
#
# QUY TẮC CẬP NHẬT (ĐÃ CHỐT):
#   - CAMERA IN : chỉ MỞ session mới khi CHƯA có session nào đang "inside".
#     Nếu đã đang "inside" -> ignore (chống tạo trùng khi đứng trước
#     camera IN nhiều frame liên tiếp).
#   - CAMERA OUT: LUÔN cập nhật, không có khái niệm "ignore":
#       + Nếu đang có session "inside" -> ĐÓNG session đó (exit_time=now,
#         status="outside") - đây là lần OUT đầu tiên sau khi vào.
#       + Nếu KHÔNG có session "inside" (đã "outside" từ trước, hoặc chưa
#         từng có session nào) -> vẫn CẬP NHẬT liên tục vào bản ghi
#         "outside" GẦN NHẤT của người đó (bump exit_time = now mỗi lần
#         thấy) thay vì bỏ qua hoàn toàn - phản ánh đúng "vẫn đang ở
#         ngoài, mới thấy lại lúc này". Cứ tiếp tục cập nhật như vậy CHO
#         TỚI KHI gặp lại camera IN thì mới tạo session MỚI (entry_time
#         mới, status="inside").
#   - Mọi document đều có "updated_at" = lần ghi/cập nhật gần nhất, dùng
#     để tìm "bản ghi gần nhất của người này" (find_latest_session) mà
#     không phụ thuộc entry_time (có thể là None với người OUT trước khi
#     từng có IN nào được ghi nhận).


def find_open_session(person_id):
    """Trả về session ĐANG MỞ (status="inside") của person_id này, hoặc
    None nếu chưa có/không có session nào đang mở."""
    return person_sessions.find_one({"person_id": person_id, "status": "inside"})


def find_latest_session(person_id):
    """Trả về bản ghi GẦN NHẤT (bất kể inside/outside) của person_id này -
    dùng để quyết định camera OUT nên đóng session đang mở hay chỉ cập
    nhật (bump) bản ghi outside gần nhất. None nếu người này chưa từng có
    bản ghi nào trong person_sessions."""
    return person_sessions.find_one(
        {"person_id": person_id}, sort=[("updated_at", -1)]
    )


def create_session(person_id, channel=None, name=None, person_type=None):
    """Tạo 1 session MỚI khi người này được camera IN nhận diện LẦN ĐẦU
    (lúc chưa có session nào đang mở cho họ) - xem handle_entry()."""
    camera = get_camera_by_channel(channel) if channel else None
    address = camera.get("address") if camera else None
    now = datetime.now()

    doc = {
        "person_id": person_id,
        "name": name,
        "type": person_type,
        "entry_time": now,
        "exit_time": None,
        "status": "inside",
        "channel_in": channel,
        "channel_out": None,
        "address": address,
        "updated_at": now,
    }
    result = person_sessions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def close_session(session, channel=None):
    """Đóng 1 session ĐANG MỞ (status='inside' -> 'outside') khi camera
    OUT nhận diện lại đúng người đó lần ĐẦU TIÊN sau khi vào."""
    now = datetime.now()
    person_sessions.update_one(
        {"_id": session["_id"]},
        {"$set": {"exit_time": now, "status": "outside", "channel_out": channel, "updated_at": now}},
    )
    session["exit_time"] = now
    session["status"] = "outside"
    session["channel_out"] = channel
    session["updated_at"] = now
    return session


def touch_exit(session, channel=None, name=None, person_type=None):
    """Cập nhật (bump) bản ghi OUTSIDE gần nhất - dùng khi camera OUT tiếp
    tục nhận diện người đó trong lúc KHÔNG có session nào đang mở (họ đã
    ở ngoài từ trước rồi). Chỉ cập nhật exit_time/channel_out/updated_at -
    KHÔNG đổi entry_time, KHÔNG tạo document mới, status vẫn "outside"."""
    now = datetime.now()
    update = {"exit_time": now, "channel_out": channel, "updated_at": now}
    if name is not None:
        update["name"] = name
    if person_type is not None:
        update["type"] = person_type

    person_sessions.update_one({"_id": session["_id"]}, {"$set": update})
    session.update(update)
    return session


def create_outside_only_record(person_id, channel=None, name=None, person_type=None):
    """Người này CHƯA TỪNG có bản ghi nào trong person_sessions nhưng vừa
    bị camera OUT thấy (vd họ vào qua cửa không có camera IN, hoặc camera
    IN từng bị lỗi lúc họ vào). Tạo 1 bản ghi "outside" với entry_time=None
    để không bịa ra thời điểm vào không có thật, nhưng vẫn có chỗ để các
    lần OUT tiếp theo tiếp tục touch_exit() cập nhật, cho tới khi họ đi
    qua camera IN thật sự thì handle_entry() sẽ mở session "inside" mới."""
    camera = get_camera_by_channel(channel) if channel else None
    address = camera.get("address") if camera else None
    now = datetime.now()

    doc = {
        "person_id": person_id,
        "name": name,
        "type": person_type,
        "entry_time": None,
        "exit_time": now,
        "status": "outside",
        "channel_in": None,
        "channel_out": channel,
        "address": address,
        "updated_at": now,
    }
    result = person_sessions.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def handle_entry(person_id, channel=None, name=None, person_type=None):
    """
    Gọi khi CAMERA IN nhận diện ra person_id này.

        session = find_open_session(person_id)
        Nếu chưa có session -> create_session()  (session MỚI)
        Nếu đã có           -> ignore (họ vẫn đang ở trong, không tạo trùng)

    Nhờ vậy dù camera IN nhận diện liên tục nhiều lần trong lúc người đó
    còn đứng trước cửa (09:00:01, 09:00:02, 09:00:03...) thì CHỈ session
    đầu tiên được tạo, các lần sau tự động bị bỏ qua vì đã có session
    "inside" - KHÔNG cần thêm cache/cooldown thời gian riêng nào khác cho
    việc này, bản thân trạng thái session đã là cơ chế chống trùng.

    Trả về session (dict) nếu vừa tạo mới, hoặc None nếu bị bỏ qua vì đã
    có session đang mở.
    """
    if find_open_session(person_id) is not None:
        return None  # đã có session "inside" -> ignore

    return create_session(person_id, channel=channel, name=name, person_type=person_type)


def handle_exit(person_id, channel=None, name=None, person_type=None):
    """
    Gọi khi CAMERA OUT nhận diện ra person_id này. KHÁC với handle_entry(),
    hàm này KHÔNG BAO GIỜ "ignore" hoàn toàn - luôn cập nhật 1 cái gì đó:

        latest = find_latest_session(person_id)   # bất kể inside/outside

        Nếu chưa từng có bản ghi nào       -> create_outside_only_record()
        Nếu bản ghi gần nhất đang "inside" -> close_session()   (lần OUT
                                               đầu tiên sau khi vào)
        Nếu bản ghi gần nhất đã "outside"  -> touch_exit()      (vẫn đang
                                               ở ngoài, chỉ cập nhật mốc
                                               thời gian thấy lại)

    CHỈ khi gặp lại camera IN, handle_entry() mới tạo session MỚI (entry_time
    mới) - đúng yêu cầu "liên tục update cam_out cho đến khi gặp cam in thì
    tạo session mới".

    Trả về bản ghi vừa cập nhật/tạo (dict).
    """
    session = find_latest_session(person_id)

    if session is None:
        return create_outside_only_record(person_id, channel=channel, name=name, person_type=person_type)

    if session["status"] == "inside":
        return close_session(session, channel=channel)

    return touch_exit(session, channel=channel, name=name, person_type=person_type)


def _day_bounds(day=None):
    """Trả về (start, end) là mốc 00:00:00 của `day` và của ngày kế tiếp
    (theo giờ server, khớp với datetime.now() dùng ở create_session()).
    day=None -> dùng hôm nay."""
    base = day or datetime.now()
    start = base.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def get_current_inside_count(address=None):
    """
    Số người HIỆN ĐANG ở trong (status="inside") - tính hoàn toàn ở tầng
    DATABASE bằng 1 lệnh count_documents(), không cần vòng lặp hay đếm gì
    ở tầng ứng dụng/camera.

        db.person_sessions.count_documents({"status": "inside"})

    address=None -> đếm gộp tất cả cơ sở. Truyền address cụ thể để chỉ
    đếm riêng 1 cơ sở.
    """
    query = {"status": "inside"}
    if address is not None:
        query["address"] = address
    return person_sessions.count_documents(query)


def get_entries_count_today(address=None, day=None):
    """
    Số LƯỢT VÀO trong ngày (mỗi session tạo mới = 1 lượt vào, đã tự loại
    trùng nhờ handle_entry()/find_open_session()).

        db.person_sessions.count_documents({
            "entry_time": {"$gte": start_day, "$lt": end_day}
        })
    """
    start, end = _day_bounds(day)
    query = {"entry_time": {"$gte": start, "$lt": end}}
    if address is not None:
        query["address"] = address
    return person_sessions.count_documents(query)


def get_unique_persons_today(address=None, day=None):
    """
    Số NGƯỜI KHÁC NHAU đã vào trong ngày (1 người vào-ra nhiều lượt trong
    ngày chỉ tính 1 lần).

        db.person_sessions.distinct("person_id", {
            "entry_time": {"$gte": start_day, "$lt": end_day}
        })
    """
    start, end = _day_bounds(day)
    query = {"entry_time": {"$gte": start, "$lt": end}}
    if address is not None:
        query["address"] = address
    return person_sessions.distinct("person_id", query)


def get_open_sessions(address=None):
    """Danh sách đầy đủ (không chỉ đếm) những người hiện đang "inside" -
    tiện cho màn hình hiển thị "ai đang ở trong" thay vì chỉ 1 con số."""
    query = {"status": "inside"}
    if address is not None:
        query["address"] = address
    return list(person_sessions.find(query).sort("entry_time", -1))


# ======================================================
# OCCUPANCY / ZONE COUNTING (camera "thấy cả 2 chiều", đếm ẨN DANH)
# ======================================================
# TÁCH BIỆT HOÀN TOÀN với face_events/persons/person_sessions phía trên -
# camera loại này (camera_role="occupancy") KHÔNG chạy face recognition,
# KHÔNG cần biết là ai, chỉ cần biết 1 track_id vừa băng qua ranh giới
# zone_in/zone_out theo hướng nào (xem zone_counter.py: ZoneCounter).
#
# 2 collection:
#   - zone_events    : lịch sử THÔ, mỗi lượt băng qua = 1 document (kèm ảnh
#                       toàn thân chụp đúng lúc băng qua) - dùng để xem lại/
#                       audit, KHÔNG dùng để tính số đếm hiện tại.
#   - zone_occupancy : SỐ NGƯỜI HIỆN ĐANG CÓ MẶT theo từng address (cơ sở),
#                       1 document/address, cập nhật bằng $inc NGUYÊN TỬ mỗi
#                       lượt vào/ra - đọc ra là có ngay, không cần aggregate.


def configure_occupancy_camera(channel, zone_in, zone_out):
    """
    Đánh dấu 1 camera ĐÃ CÓ trong `cameras` là camera "occupancy" (thấy cả
    2 chiều, đếm ẩn danh) + lưu polygon zone_in/zone_out của nó.

    zone_in / zone_out: list [(x, y), ...] - toạ độ điểm THEO ĐÚNG kích
    thước khung hình của camera này (lấy bằng configure_zone_camera.py).

    KHÔNG đụng tới field "is_in" - is_in chỉ có ý nghĩa với camera
    "identity" (cố định 1 chiều). camera_role là field RIÊNG để multi_main.py
    biết chạy CameraPipeline (identity) hay ZoneCameraPipeline (occupancy)
    cho camera này (xem load_identity_cameras_from_db/load_occupancy_cameras_from_db).
    """
    return cameras.update_one(
        {"channel": channel},
        {"$set": {
            "camera_role": "occupancy",
            "zone_in": [list(p) for p in zone_in],
            "zone_out": [list(p) for p in zone_out],
        }},
    )


def get_occupancy_cameras():
    """Camera đang bật (status=True) VÀ đã được cấu hình camera_role=occupancy
    - dùng để multi_main.py biết camera nào chạy ZoneCameraPipeline."""
    return list(cameras.find({"status": True, "camera_role": "occupancy"}))


def get_identity_cameras():
    """Camera đang bật (status=True) dùng cho nhận diện danh tính - tức
    camera_role KHÁC 'occupancy' (bao gồm camera_role='identity' HOẶC các
    camera CŨ chưa từng có field camera_role - tương thích ngược, coi như
    mặc định là 'identity')."""
    return list(cameras.find({
        "status": True,
        "camera_role": {"$ne": "occupancy"},
    }))


def save_zone_event(channel, direction, image_b64=None, track_id=None):
    """
    Ghi lại 1 lượt VÀO/RA ẨN DANH vừa phát hiện bởi camera occupancy.

    direction: "in" hoặc "out"

    Đồng thời cập nhật LUÔN zone_occupancy (số người đang có mặt) theo
    address của camera này bằng $inc - "in" thì +1, "out" thì -1. Nếu ghi
    zone_events (lịch sử ảnh) lỗi (Mongo tạm mất kết nối), vẫn CỐ GẮNG cập
    nhật số đếm (quan trọng hơn ảnh lịch sử) - 2 việc không phụ thuộc nhau.
    """
    camera = get_camera_by_channel(channel)
    address = camera.get("address") if camera else None

    doc = {
        "channel": channel,
        "address": address,
        "direction": direction,
        "track_id": track_id,
        "image": image_b64,
        "time": datetime.now(),
    }

    try:
        zone_events.insert_one(doc)
    except PyMongoError as e:
        logger.error(
            "[%s] Ghi zone_event lên Mongo thất bại (%s) - chỉ mất lịch sử "
            "ảnh lượt này, KHÔNG ảnh hưởng số đếm hiện tại (vẫn cập nhật bên dưới)",
            channel, e,
        )

    if address is None:
        logger.warning(
            "[%s] Camera occupancy thiếu 'address' trong DB - không xác định "
            "được đếm vào zone_occupancy của cơ sở nào, bỏ qua cập nhật số đếm.",
            channel,
        )
        return

    delta = 1 if direction == "in" else -1
    try:
        zone_occupancy.update_one(
            {"address": address},
            {"$inc": {"count": delta}, "$set": {"updated_at": datetime.now()}},
            upsert=True,
        )
        # Không để count âm (vd lệch số hiếm gặp do bỏ sót lượt ra/vào khi
        # camera mất kết nối) - clamp về 0 thay vì để âm vô lý.
        zone_occupancy.update_one(
            {"address": address, "count": {"$lt": 0}},
            {"$set": {"count": 0}},
        )
    except PyMongoError as e:
        logger.error("Cập nhật zone_occupancy thất bại (address=%s): %s", address, e)


def get_zone_occupancy_count(address=None):
    """Số người HIỆN ĐANG có mặt tính theo đếm zone (occupancy camera) -
    đọc thẳng field 'count' đã được $inc sẵn từ trước, KHÔNG cần aggregate
    lại mỗi lần gọi (khác get_current_staying_count() ở trên vốn phải
    aggregate face_events).

    address=None -> cộng dồn TẤT CẢ cơ sở."""
    if address is not None:
        doc = zone_occupancy.find_one({"address": address})
        return doc["count"] if doc else 0

    total = 0
    for doc in zone_occupancy.find({}):
        total += doc.get("count", 0)
    return total


def reset_zone_occupancy(address, count=0):
    """Đặt lại số đếm thủ công (vd đầu ngày, hoặc khi phát hiện lệch số do
    camera mất kết nối/bỏ sót lượt ra-vào) - camera occupancy (ẩn danh)
    KHÔNG có cách nào tự phát hiện + tự sửa lệch số như cơ chế
    person_sessions (có danh tính ổn định để đối chiếu)."""
    zone_occupancy.update_one(
        {"address": address},
        {"$set": {"count": count, "updated_at": datetime.now()}},
        upsert=True,
    )


def get_zone_events(channel=None, limit=100):
    """Lịch sử thô các lượt vào/ra ẩn danh (kèm ảnh) - dùng để xem lại/audit,
    KHÔNG dùng để tính số đếm hiện tại (xem get_zone_occupancy_count)."""
    query = {"channel": channel} if channel else {}
    return list(zone_events.find(query).sort("time", -1).limit(limit))


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
    address,
    url=None,
    ip=None,
    status=True,
    recorder_id=None,
    paired_channel=None,
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
    paired_channel -> channel của camera CẶP với camera này (cùng 1 cửa,
                   hướng ngược lại - vd 1 IN + 1 OUT cùng 1 cửa). Chỉ có ý
                   nghĩa nếu bạn dùng cơ chế reset dedup theo camera cặp ở
                   tầng RecognitionWorker; camera KHÔNG thuộc cặp nào thì để
                   None (mặc định). Field này KHÔNG được ZoneCameraPipeline/
                   camera_role="occupancy" sử dụng - occupancy dùng
                   zone_in/zone_out (xem configure_occupancy_camera()).
    """
    doc = {
        "serial": serial,
        "channel": channel,
        "is_in": is_in,
        "url": url,
        "ip": ip,
        "address": address,
        "status": status,
        "recorder_id": ObjectId(recorder_id) if recorder_id else None,
        "paired_channel": paired_channel,
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
    import matplotlib.pyplot as plt  # import cục bộ - xem ghi chú ở đầu file
    rows = math.ceil(len(events) / cols)
    plt.figure(figsize=(cols * 4, rows * 4))

    for i, ev in enumerate(events):
        img = base64_to_pil(ev["image"])
        img = draw_info_on_image(
            img,
            ev.get("name", "Unknown"),
            ev["time"],
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
    events = get_zone_events()
    # print(events)
    show_face_images(events)
    # print(get_all_persons())
    # print(get_current_staying_count(address="cam1"))
    
    
    
    
    