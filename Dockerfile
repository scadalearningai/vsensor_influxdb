FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=Asia/Jakarta

# System dependencies:
# - libgomp1: OpenMP runtime required by LightGBM & XGBoost
# - unixodbc: ODBC driver manager required by pyodbc (SQL Server)
# - cron & tzdata: 5-minute scheduled inference with local timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    unixodbc \
    cron \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY fuel_gas/ ./fuel_gas/
COPY permeate_sensor/ ./permeate_sensor/
COPY docker/ ./docker/

# Normalize line endings (CRLF -> LF) for Linux scripts & crontab
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh /app/docker/crontab \
    && chmod +x /app/docker/entrypoint.sh \
    && cp /app/docker/crontab /etc/cron.d/vsensor-cron \
    && chmod 0644 /etc/cron.d/vsensor-cron \
    && crontab /etc/cron.d/vsensor-cron

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["cron"]
