#!/usr/bin/env bash
set -e

if [ "$1" = "cron" ]; then
    # Export container environment variables so cron jobs can read INFLUX_*, TIMESCALE_*, and DB settings
    printenv | grep -E '^(INFLUX_|TIMESCALE_|TZ=|PYTHONPATH=)' > /etc/environment || true

    touch /var/log/permeate_inference.log /var/log/fuelgas_inference.log
    cron
    echo "[vsensor-cron] Cron daemon started (permeate_sensor & fuel_gas every 5 min at :30s)."
    exec tail -F /var/log/permeate_inference.log /var/log/fuelgas_inference.log
fi

exec "$@"
