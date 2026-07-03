# Đóng gói dự án thành binary Linux (PyInstaller)

## 1. Cài PyInstaller vào ĐÚNG env đang chạy project

```bash
cd /home/lychien/Desktop/Project_new
pip install pyinstaller
```

## 2. Build

```bash
chmod +x build_multi_main.sh build_manage_persons.sh
./build_multi_main.sh
./build_manage_persons.sh
```

Kết quả nằm ở `dist/multi_main/` và `dist/manage_persons/` — mỗi thư mục là
1 bản đóng gói ĐỘC LẬP (có thể copy cả thư mục sang máy Linux khác chạy,
không cần cài Python/torch/opencv... ở máy đó).

## 3. Trước khi chạy trên máy đích (checklist)

- [ ] Copy **cả thư mục** `dist/<tên_app>/`, không phải chỉ file thực thi.
- [ ] Copy `~/.insightface/` (model đã tải sẵn ở máy dev) sang máy đích,
      hoặc set `INSIGHTFACE_HOME=/đường/dẫn/bạn/copy/tới` trước khi chạy.
      Nếu không, lần chạy đầu insightface sẽ cố tải model qua Internet.
- [ ] Set `MONGO_URI` qua biến môi trường, không hardcode vào máy đích:
      `export MONGO_URI="mongodb://user:pass@host/db"`
- [ ] Máy đích có môi trường đồ hoạ (X11/Wayland) nếu chạy `multi_main`
      (dùng `cv2.imshow`). Nếu là server không màn hình, cần chạy qua
      Xvfb hoặc chuyển hướng hiển thị (X11 forwarding qua SSH), hoặc sửa
      code bỏ `cv2.imshow`/`cv2.waitKey` để chạy hoàn toàn headless.
- [ ] Máy đích thiếu thư viện hệ thống mà OpenCV cần (thường gặp trên
      server tối giản / Docker `slim`):
      ```bash
      sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
      ```
- [ ] Luôn `cd` vào đúng `dist/<tên_app>/` rồi mới chạy binary (code dùng
      đường dẫn tương đối như `detectors/yolov8n.pt`, `saved_person/...`
      tính theo thư mục làm việc hiện tại — chạy từ nơi khác sẽ lỗi
      "không tìm thấy file").

## 4. Build trên đúng máy/distro đích

Binary Linux được build ra sẽ **link động (dynamic link)** với glibc và một
số thư viện hệ thống của máy build. Nếu máy đích chạy distro/glibc cũ hơn
máy build, binary có thể báo lỗi kiểu `GLIBC_2.XX not found`. Cách an toàn
nhất:

- Build **trực tiếp trên máy đích** (hoặc 1 máy/VM cùng distro & version), hoặc
- Build trong container Docker dùng image base **giống hệt hoặc cũ hơn**
  distro của máy đích (ví dụ `ubuntu:20.04` nếu máy đích cũng Ubuntu 20.04
  hoặc mới hơn).

## 5. Vì sao dùng `--onedir` thay vì `--onefile`

`--onefile` nén mọi thứ vào 1 file, nhưng MỖI LẦN CHẠY sẽ tự giải nén lại
ra thư mục tạm (`/tmp/_MEIxxxxx`) — với torch/ultralytics/insightface nặng
cỡ 1-2GB, việc này khiến mỗi lần khởi động chậm thêm nhiều giây. `--onedir`
giải nén 1 lần khi build, chạy lên là dùng ngay.

## 6. Kích thước dự kiến

Vì có torch + ultralytics + insightface + faiss + opencv, thư mục
`dist/multi_main/` có thể nặng **1.5 - 3GB**. Đây là điều bình thường với
stack ML này, không phải lỗi build.
