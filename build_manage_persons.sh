#!/usr/bin/env bash
# ==========================================================
# Build manage_persons.py -> binary Linux (chế độ --onedir)
#
# Nhẹ hơn multi_main vì không cần torch/ultralytics (manage_persons.py chỉ
# dùng insightface để trích embedding + pymongo để ghi DB).
#
# CHẠY SCRIPT NÀY:
#   - Từ thư mục GỐC của project
#   - Trong đúng virtualenv/conda env đang có opencv, insightface, pymongo
#   - Trên đúng máy Linux (hoặc distro/glibc tương đương) với máy sẽ CHẠY
#     binary sau này
#
# CÁCH DÙNG:
#   chmod +x build_manage_persons.sh
#   ./build_manage_persons.sh
# ==========================================================

set -e

APP_NAME="manage_persons"

pip install --upgrade pyinstaller

pyinstaller \
    --name "$APP_NAME" \
    --onedir \
    --noconfirm \
    --clean \
    --collect-all insightface \
    --collect-data cv2 \
    --hidden-import pymongo \
    --hidden-import bson \
    manage_persons.py

echo ""
echo "=================================================="
echo "Build xong."
echo "Thư mục kết quả: dist/${APP_NAME}/"
echo ""
echo "QUAN TRỌNG - trước khi chạy trên máy đích:"
echo "1. Copy CẢ THƯ MỤC dist/${APP_NAME}/ sang máy đích."
echo "2. Copy thêm thư mục model insightface: ~/.insightface/"
echo "   sang đúng đường dẫn ~/.insightface/ ở máy đích (hoặc set biến"
echo "   môi trường INSIGHTFACE_HOME), nếu không insightface sẽ cố tải"
echo "   model từ Internet ở lần chạy đầu."
echo "3. Set biến môi trường MONGO_URI trước khi chạy."
echo "4. Đây là công cụ CLI (argparse) - chạy y hệt cú pháp cũ, ví dụ:"
echo "     cd dist/${APP_NAME}"
echo "     MONGO_URI='mongodb://...' ./${APP_NAME} list"
echo "     MONGO_URI='mongodb://...' ./${APP_NAME} add --name \"Nhung\" --webcam"
echo "=================================================="
