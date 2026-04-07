# Sử dụng Python 3.11 slim để nhẹ máy và ổn định
FROM python:3.11-slim

# 1. Cài đặt các thư viện hệ thống cần thiết cho OpenCV, Pillow và AI Models
# Thay libgl1-mesa-glx bằng libgl1 (Gói chuẩn cho bản Debian mới)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc trong container
WORKDIR /app

# 3. Copy file requirements trước để tận dụng Docker Cache
# Giúp các lần build sau nhanh hơn nếu bạn không thay đổi thư viện
COPY requirements.txt .

# 4. Nâng cấp pip và cài đặt thư viện Python
# Sử dụng --no-cache-dir để giảm dung lượng image Docker
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ mã nguồn vào container
COPY . .

# 6. Mở cổng 8000 cho FastAPI
EXPOSE 8000

# 7. Lệnh khởi chạy server (Lưu ý: app.main:app tùy thuộc vào cấu trúc code của bạn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]