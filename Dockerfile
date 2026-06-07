# WEXON T2 I2V GPU worker — LTX-Video (OpenRAIL-M, коммерция разрешена)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Модель LTX качается при первом запросе в HF-кэш (для прода — network volume).
# HF_HOME можно переопределить env-переменной эндпоинта на путь network volume.
ENV HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    LTX_MODEL=Lightricks/LTX-Video

COPY handler.py .

CMD ["python", "-u", "handler.py"]
