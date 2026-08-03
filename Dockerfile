# Sử dụng base image Python chính thức, gọn nhẹ
FROM python:3.12-slim

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cài đặt các gói phụ thuộc hệ thống cần thiết cho biên dịch (nếu có)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cơ chế caching của Docker
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn ứng dụng vào container
COPY . .

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]