FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# system deps (ffmpeg + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the full Render backend dependencies, including OpenClip and Whisper.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy app, OpenClip modules and prompt templates
COPY app /app/app
COPY openclip_core /app/openclip_core
COPY prompts /app/prompts

# create storage dir
RUN mkdir -p /storage /data

# Render provides PORT at runtime.
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
