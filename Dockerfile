Copy code
# Sử dụng image Python chính thức
FROM python:3.10-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements vào container
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY . .

# Đặt lệnh mặc định khi container khởi động (không dùng main.py)
CMD ["python", "app.py"]