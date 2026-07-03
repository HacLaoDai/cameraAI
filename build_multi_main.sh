#!/usr/bin/env bash
# ==========================================================
# Build multi_main.py -> binary Linux (chế độ --onedir)
#
# CHẠY SCRIPT NÀY:
#   - Từ thư mục GỐC của project (nơi có multi_main.py, camera/, detectors/,
#     recognition/, trackers/, database/ ...)
#   - Trong đúng virtualenv/conda env đang có sẵn opencv, torch, ultralytics,
#     insightface, faiss, pymongo, pillow, matplotlib (env bạn đang code)
#   - Trên đúng máy Linux (hoặc distro/glibc tương đương) với máy sẽ CHẠY
#     binary sau này - binary Linux không portable tuỳ ý giữa các distro.
#
# CÁCH DÙNG:
#   chmod +x build_multi_main.sh
#   ./build_multi_main.sh
# ==========================================================

set -e

APP_NAME="multi_main"

pip install --upgrade pyinstaller

pyinstaller \
    --name "$APP_NAME" \
    --onedir \
    --noconfirm \
    --clean \
    --collect-all ultralytics \
    --collect-all insightface \
    --collect-all faiss \
    --collect-data cv2 \
    --hidden-import pymongo \
    --hidden-import bson \
    --hidden-import PIL \
    --hidden-import matplotlib \
    --add-data "detectors/yolov8n.pt:detectors" \
    multi_main.py

echo ""
echo "=================================================="
echo "Build xong."
echo "Thư mục kết quả: dist/${APP_NAME}/"
echo ""
echo "QUAN TRỌNG - trước khi chạy trên máy đích:"
echo "1. Copy CẢ THƯ MỤC dist/${APP_NAME}/ sang máy đích (không phải chỉ"
echo "   file thực thi bên trong, vì nó cần các .so/.pt/.yaml đi kèm)."
echo "2. Copy thêm thư mục model insightface: ~/.insightface/"
echo "   sang đúng đường dẫn ~/.insightface/ ở máy đích (hoặc set biến"
echo "   môi trường INSIGHTFACE_HOME trỏ tới nơi bạn copy tới), nếu không"
echo "   insightface sẽ cố tải model từ Internet ở lần chạy đầu."
echo "3. Set biến môi trường MONGO_URI trước khi chạy (không hardcode)."
echo "4. LUÔN cd vào đúng thư mục dist/${APP_NAME}/ rồi mới chạy binary,"
echo "   vì code dùng đường dẫn tương đối (detectors/yolov8n.pt,"
echo "   saved_person/...) tính theo thư mục làm việc hiện tại."
echo ""
echo "Chạy thử:"
echo "  cd dist/${APP_NAME}"
echo "  MONGO_URI='mongodb://...' ./${APP_NAME}"
echo "=================================================="
