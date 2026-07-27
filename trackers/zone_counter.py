import cv2
import numpy as np


class ZoneCounter:
    """
    Đếm người ra/vào ẨN DANH (không cần biết là ai) cho camera "thấy cả 2
    chiều" (1 camera quan sát 1 cửa, người có thể đi cả 2 hướng qua đó) -
    chuyển thể từ logic gốc trong main.py (2 polygon area1/area2 + dict
    enter/exits + list1/list2 đếm dedup theo track_id), viết lại thành 1
    class tái sử dụng được cho nhiều camera - mỗi camera 1 instance riêng,
    KHÔNG share state với nhau.

    KHÁC với PersonTracker (dùng cho camera nhận diện danh tính): class
    này KHÔNG quan tâm khuôn mặt là ai, CHỈ quan tâm 1 track_id đã băng
    qua đủ 2 vùng theo đúng thứ tự nào:
        zone_out -> zone_in  = 1 lượt VÀO ("in")
        zone_in  -> zone_out = 1 lượt RA  ("out")
    để đếm đúng 1 lần/lượt, không đếm trùng khi track đứng lảng vảng ngay
    biên giới 2 vùng nhiều frame liên tiếp.

    zone_in / zone_out: list các điểm (x, y) tạo thành polygon, cùng format
    area1/area2 trong main.py gốc - lấy bằng công cụ chọn điểm bằng chuột
    (xem configure_zone_camera.py).
    """

    def __init__(self, zone_in, zone_out, point_mode="bottom_center"):
        self.zone_in = np.array(zone_in, dtype=np.int32)
        self.zone_out = np.array(zone_out, dtype=np.int32)
        self.point_mode = point_mode

        # track_id đã thấy đứng trong zone_out ("đang chờ băng vào") / zone_in
        # ("đang chờ băng ra") - dùng để xác định ĐÚNG THỨ TỰ băng qua, không
        # chỉ đơn thuần "đang đứng trong 1 vùng nào đó".
        self._pending_in = {}   # track_id -> True
        self._pending_out = {}  # track_id -> True

        # Dedup: track_id đã được TÍNH là "đã vào"/"đã ra" (giống list1/list2
        # trong main.py) - tránh đếm lặp nếu track dao động qua lại đúng biên
        # nhiều frame. Được dọn khi track biến mất lâu (xem _cleanup).
        self._counted_in = set()
        self._counted_out = set()

        self._last_seen_frame = {}  # track_id -> frame_index gần nhất còn thấy

    def _point_for(self, bbox):
        x1, y1, x2, y2 = bbox
        if self.point_mode == "bottom_left":
            # Giữ lại lựa chọn điểm giống HỆT main.py gốc (x1, y2) nếu cần
            # so sánh/tương thích ngược.
            return (x1, y2)
        # Mặc định: điểm đáy-giữa bbox - ổn định hơn bottom-left của bản gốc
        # khi người đi chéo hướng hoặc camera đặt nghiêng.
        return ((x1 + x2) // 2, y2)

    def update(self, detections, frame_index, stale_after=90):
        """
        detections: list [{"track_id":, "bbox": [x1,y1,x2,y2]}, ...] của
        frame hiện tại (đã có track_id ổn định từ ByteTrack, giống format
        PersonDetector.detect() trả về).

        Trả về list SỰ KIỆN MỚI phát sinh ở đúng frame này (rỗng nếu không
        ai vừa hoàn tất 1 lượt băng qua):
            [{"track_id":, "direction": "in"|"out", "bbox": [x1,y1,x2,y2]}]
        """
        events = []
        active_ids = set()

        for det in detections:
            track_id = det["track_id"]
            bbox = det["bbox"]
            active_ids.add(track_id)
            self._last_seen_frame[track_id] = frame_index

            point = self._point_for(bbox)

            in_zone_in = cv2.pointPolygonTest(self.zone_in, point, False) >= 0
            in_zone_out = cv2.pointPolygonTest(self.zone_out, point, False) >= 0

            # ---- Hướng VÀO: phải thấy ở zone_out TRƯỚC, rồi mới băng vào zone_in ----
            if in_zone_out:
                self._pending_in[track_id] = True

            if in_zone_in and self._pending_in.get(track_id):
                if track_id not in self._counted_in:
                    self._counted_in.add(track_id)
                    events.append({"track_id": track_id, "direction": "in", "bbox": bbox})
                self._pending_in.pop(track_id, None)

            # ---- Hướng RA: phải thấy ở zone_in TRƯỚC, rồi mới băng ra zone_out ----
            if in_zone_in:
                self._pending_out[track_id] = True

            if in_zone_out and self._pending_out.get(track_id):
                if track_id not in self._counted_out:
                    self._counted_out.add(track_id)
                    events.append({"track_id": track_id, "direction": "out", "bbox": bbox})
                self._pending_out.pop(track_id, None)

        self._cleanup(active_ids, frame_index, stale_after)
        return events

    def _cleanup(self, active_ids, frame_index, stale_after):
        """Dọn state của các track_id đã biến mất lâu (không còn trong
        active_ids) để RAM không phình vô hạn khi chạy 24/7, và để tránh
        1 track_id (sẽ bị ByteTrack tái sử dụng cho người MỚI sau này) bị
        dính dedup của người cũ mãi mãi."""
        stale = [
            tid for tid, last in self._last_seen_frame.items()
            if frame_index - last > stale_after
        ]
        for tid in stale:
            self._last_seen_frame.pop(tid, None)
            self._pending_in.pop(tid, None)
            self._pending_out.pop(tid, None)
            self._counted_in.discard(tid)
            self._counted_out.discard(tid)

    def draw(self, frame, entered_total, exited_total, current_override=None):
        """Vẽ 2 vùng polygon + HUD Entered/Exited/Current lên frame - tương
        đương phần cvzone.putTextRect trong main.py gốc (đổi sang cv2.putText
        thuần vì pipeline hiện tại không phụ thuộc cvzone).

        current_override: nếu truyền vào (khác None), dùng giá trị này làm
        "Current" thay vì tự tính entered_total - exited_total. Cần thiết
        khi entered_total được cộng thêm 1 baseline lúc khởi động (số người
        đang có mặt lấy từ DB - xem ZoneCameraPipeline.__init__ trong
        multi_main.py) - lúc đó entered_total không còn thuần là "số lượt
        vào trong phiên chạy này" nữa, nhưng entered_total - exited_total
        vẫn LUÔN đúng bằng số đang có mặt thật, nên vẫn dùng được làm mặc định
        khi không truyền current_override."""
        cv2.polylines(frame, [self.zone_in], True, (0, 255, 255), 2)
        cv2.polylines(frame, [self.zone_out], True, (255, 0, 255), 2)

        current = entered_total - exited_total if current_override is None else current_override
        cv2.putText(frame, f"Entered: {entered_total}", (1000, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Exited: {exited_total}", (1000, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Current: {current}", (1000, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)