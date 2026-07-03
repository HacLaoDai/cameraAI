#!/usr/bin/env bash
# ==========================================================
# Build manage_cameras.py -> binary Linux (chế độ --onedir)
# Nhẹ nhất trong 3 app: chỉ cần pymongo, không cần torch/insightface.
#
# Chạy từ thư mục GỐC project, trong đúng venv đang có pymongo.
# ==========================================================

set -e

APP_NAME="manage_cameras"

pip install --upgrade pyinstaller

pyinstaller \
    --name "$APP_NAME" \
    --onedir \
    --noconfirm \
    --clean \
    --hidden-import pymongo \
    --hidden-import bson \
    manage_cameras.py

echo ""
echo "=================================================="
echo "Build xong. Thư mục kết quả: dist/${APP_NAME}/"
echo "Set MONGO_URI trước khi chạy, ví dụ:"
echo "  cd dist/${APP_NAME}"
echo "  MONGO_URI='mongodb://...' ./${APP_NAME} list"
echo "=================================================="
