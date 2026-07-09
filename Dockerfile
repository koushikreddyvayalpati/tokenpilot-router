FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY app ./app
COPY eval ./eval

ENV PYTHONPATH=/app
RUN mkdir -p /input /output
ENTRYPOINT ["python", "-m", "app.submission"]
