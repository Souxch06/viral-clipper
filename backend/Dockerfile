FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# system deps (ffmpeg + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY app /app/app

# create storage dir
RUN mkdir -p /storage /data

# expose uvicorn port
EXPOSE 8000

# default entry: run uvicorn (overridden in docker-compose for worker)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
