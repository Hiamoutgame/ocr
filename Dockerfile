FROM python:3.10-slim

WORKDIR /app

# Cài lib hệ thống cần cho OCR/PDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước (tối ưu cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

CMD ["python", "main.py"]