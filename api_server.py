"""
API Server cho hệ thống camera pipeline - quản lý persons, cameras, tra cứu
sự kiện ra/vào. KHÔNG gọi trực tiếp vào main_multi.py - giao tiếp gián tiếp
qua MongoDB dùng chung (task_db.py). Nhờ vậy API server có thể chạy trên máy
khác hoàn toàn với máy chạy pipeline camera, miễn là cùng trỏ vào 1 MongoDB.

CÀI ĐẶT (nếu chưa có):
    pip install fastapi uvicorn python-multipart --break-system-packages

CHẠY:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Sau khi chạy, mở http://<ip-server>:8000/docs để xem giao diện test API
tự động sinh ra (Swagger UI) - không cần Postman.
"""

from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from database import task_db
from detectors.detect_face import ArcFaceExtractor

app = FastAPI(title="Camera Pipeline API")


# ======================================================
# FACE EXTRACTOR - load 1 lần, dùng chung cho mọi request enroll/edit
# ======================================================
_face_extractor = None


def get_face_extractor():
    global _face_extractor
    if _face_extractor is None:
        _face_extractor = ArcFaceExtractor(ctx_id=-1)  # -1 = CPU
    return _face_extractor


def get_largest_embedding(img):
    extractor = get_face_extractor()
    embeddings, bboxes = extractor.extract_embeddings(img)

    if len(embeddings) == 0:
        return None

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    idx = areas.index(max(areas))
    return embeddings[idx]


async def files_to_embeddings(files: List[UploadFile]):
    embeddings = []
    skipped = []

    for f in files:
        content = await f.read()
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            skipped.append(f.filename)
            continue

        emb = get_largest_embedding(img)

        if emb is None:
            skipped.append(f.filename)
            continue

        embeddings.append(emb)

    return embeddings, skipped


# ======================================================
# PERSONS
# ======================================================
class PersonUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    sex: Optional[int] = None
    age: Optional[int] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    nation: Optional[str] = None


def serialize_person(p):
    return {
        "person_id": p.get("person_id"),
        "name": p.get("name"),
        "type": p.get("type"),
        "sex": p.get("sex"),
        "age": p.get("age"),
        "phone": p.get("phone"),
        "email": p.get("email"),
        "nation": p.get("nation"),
        "num_embeddings": len(p.get("embeddings", [])),
    }


@app.get("/persons")
def list_persons():
    return [serialize_person(p) for p in task_db.get_all_persons()]


@app.get("/persons/{person_id}")
def get_person(person_id: int):
    p = task_db.find_person_by_person_id(person_id)
    if not p:
        raise HTTPException(404, "Không tìm thấy person_id này")
    return serialize_person(p)


@app.post("/persons")
async def create_person(
    name: str = Form(...),
    type: str = Form("nhan_vien"),
    sex: int = Form(0),
    age: int = Form(0),
    phone: str = Form(""),
    email: str = Form(""),
    nation: str = Form("VN"),
    images: List[UploadFile] = File(...),
):
    """Tạo người mới (hoặc thêm embedding nếu name đã tồn tại). Cần >=1 ảnh."""
    embeddings, skipped = await files_to_embeddings(images)

    if not embeddings:
        raise HTTPException(
            400, f"Không trích được embedding nào từ ảnh đã gửi. Bỏ qua: {skipped}"
        )

    existing = task_db.find_person_by_name(name)

    if existing:
        person_id = existing["person_id"]
        for emb in embeddings:
            task_db.add_embedding_to_person(person_id, emb)
        created = False
    else:
        info = {
            "name": name, "type": type, "sex": sex, "age": age,
            "phone": phone, "email": email, "nation": nation,
        }
        first, rest = embeddings[0], embeddings[1:]
        person_id = task_db.insert_new_person(info, first)
        for emb in rest:
            task_db.add_embedding_to_person(person_id, emb)
        created = True

    return {
        "person_id": person_id,
        "created": created,
        "embeddings_added": len(embeddings),
        "skipped_files": skipped,
    }


@app.put("/persons/{person_id}")
def update_person(person_id: int, data: PersonUpdate):
    """Sửa thông tin (không đụng vào embedding)."""
    existing = task_db.find_person_by_person_id(person_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy person_id này")

    update_data = {k: v for k, v in data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "Không có field nào để cập nhật")

    task_db.update_person(person_id, update_data)
    return {"person_id": person_id, "updated": update_data}


@app.post("/persons/{person_id}/embeddings")
async def add_embeddings(person_id: int, images: List[UploadFile] = File(...)):
    """Thêm embedding bổ sung (KHÔNG xóa embedding cũ)."""
    existing = task_db.find_person_by_person_id(person_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy person_id này")

    embeddings, skipped = await files_to_embeddings(images)

    if not embeddings:
        raise HTTPException(400, f"Không trích được embedding nào. Bỏ qua: {skipped}")

    for emb in embeddings:
        task_db.add_embedding_to_person(person_id, emb)

    return {"person_id": person_id, "embeddings_added": len(embeddings), "skipped_files": skipped}


@app.delete("/persons/{person_id}/embeddings")
def clear_embeddings(person_id: int):
    """Xóa hết embedding cũ (dùng khi muốn enroll lại từ đầu)."""
    existing = task_db.find_person_by_person_id(person_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy person_id này")

    task_db.clear_person_embeddings(person_id)
    return {"person_id": person_id, "cleared": True}


@app.delete("/persons/{person_id}")
def delete_person(person_id: int):
    """Xóa hẳn 1 người."""
    existing = task_db.find_person_by_person_id(person_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy person_id này")

    task_db.delete_person(person_id)
    return {"person_id": person_id, "deleted": True}


# ======================================================
# CAMERAS
# ======================================================
class CameraCreate(BaseModel):
    serial: str
    channel: str          # PHẢI trùng với "name" khai báo trong CAMERAS ở main_multi.py
    is_in: bool            # True = camera cổng vào, False = camera cổng ra
    ip: Optional[str] = None
    status: bool = True
    recorder_id: Optional[str] = None


class CameraUpdate(BaseModel):
    serial: Optional[str] = None
    channel: Optional[str] = None
    is_in: Optional[bool] = None
    ip: Optional[str] = None
    status: Optional[bool] = None
    recorder_id: Optional[str] = None


def serialize_camera(c):
    return {
        "id": str(c["_id"]),
        "serial": c.get("serial"),
        "channel": c.get("channel"),
        "is_in": c.get("is_in"),
        "ip": c.get("ip"),
        "address":c.get("address"),
        "url": c.get("url"),
        "created_at": c.get("created_at"),
        "status": c.get("status"),
        "recorder_id": str(c["recorder_id"]) if c.get("recorder_id") else None,
    }


@app.get("/cameras")
def list_cameras():
    return [serialize_camera(c) for c in task_db.get_all_cameras()]


@app.get("/cameras/{camera_id}")
def get_camera(camera_id: str):
    c = task_db.get_camera_by_id(camera_id)
    if not c:
        raise HTTPException(404, "Không tìm thấy camera")
    return serialize_camera(c)


@app.post("/cameras")
def create_camera(data: CameraCreate):
    existing = task_db.get_camera_by_serial(data.serial)
    if existing:
        raise HTTPException(400, f"Serial '{data.serial}' đã tồn tại")

    camera_id = task_db.create_camera(
        serial=data.serial,
        channel=data.channel,
        is_in=data.is_in,
        ip=data.ip,
        status=data.status,
        recorder_id=data.recorder_id,
    )
    return {"id": str(camera_id)}


@app.put("/cameras/{camera_id}")
def update_camera(camera_id: str, data: CameraUpdate):
    existing = task_db.get_camera_by_id(camera_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy camera")

    update_data = {k: v for k, v in data.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "Không có field nào để cập nhật")

    task_db.update_camera(camera_id, update_data)
    return {"id": camera_id, "updated": update_data}


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: str):
    existing = task_db.get_camera_by_id(camera_id)
    if not existing:
        raise HTTPException(404, "Không tìm thấy camera")

    task_db.delete_camera(camera_id)
    return {"id": camera_id, "deleted": True}


# ======================================================
# EVENTS - tra cứu ra/vào
# ======================================================
@app.get("/events")
def list_events(limit: int = 50):
    events = task_db.get_face_events_simple(limit=limit)
    for e in events:
        e["_id"] = str(e["_id"])
    return events


@app.get("/events/current-staying")
def current_staying():
    """Danh sách người hiện đang 'ở trong' (event gần nhất thuộc camera is_in=True)."""
    events = task_db.get_current_staying()
    for e in events:
        e["_id"] = str(e["_id"])
    return events


@app.get("/events/person/{person_id}")
def person_movements(person_id):
    return task_db.get_person_movements(person_id)


# ======================================================
# HEALTH CHECK
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}
