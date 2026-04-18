FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install fastapi uvicorn google-genai pysubs2 
RUN pip install ffsubsync "setuptools<=81"
COPY main.py .
# Create the data folder
RUN mkdir -p /app/data && chmod 777 /app/data
EXPOSE 8000
