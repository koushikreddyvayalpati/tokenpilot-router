FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY eval ./eval

ENV PYTHONPATH=/app
RUN mkdir -p /input /output
ENTRYPOINT ["python", "-m", "app.submission"]
