# Dùng Ubuntu bản nhẹ
FROM ubuntu:22.04

# Cập nhật và cài Python + pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy toàn bộ mã nguồn
COPY . .

# Cài thư viện từ requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Mở cổng Streamlit
EXPOSE 8501

# Chạy app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
