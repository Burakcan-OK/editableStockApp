# Python 3.10 tabanlı imaj
FROM python:3.10-slim

# Çalışma klasörü
WORKDIR /app

# Gereksinimler
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Çalıştırılacak komut
CMD ["python", "main.py"]
