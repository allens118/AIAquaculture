import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

BROKER = "jetsion.com"  # MQTT broker hostname or IP
PORT = 1883
KEEPALIVE = 60

TOPICS = [
    "jetsion/30AEA4903C20/WaterPH",
    "jetsion/30AEA4903C20/WaterO2",
    "jetsion/30AEA4903C20/WaterTemp",
]
OUT_CSV = Path("aquaculture_stream.csv")
HEADERS = ["ts", "iso_time", "ph", "o2", "temp"]

buffer = {"ph": None, "o2": None, "temp": None}
topic_map = {
    TOPICS[0]: "ph",
    TOPICS[1]: "o2",
    TOPICS[2]: "temp",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def ensure_csv_header(path: Path) -> None:
    """Make sure CSV exists and has the expected header."""
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="ascii") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)
        return

    with path.open("r", newline="", encoding="ascii") as file:
        reader = csv.reader(file)
        current_header = next(reader, None)

    if current_header == HEADERS:
        return

    logging.info("Upgrading existing CSV header to include readable time")
    rows = []
    with path.open("r", newline="", encoding="ascii") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ts_str = row.get("ts")
            if ts_str is None:
                continue
            try:
                timestamp_ms = int(float(ts_str))
            except ValueError:
                logging.warning("Skipping row with invalid ts: %r", ts_str)
                continue
            iso_time = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()
            rows.append(
                {
                    "ts": timestamp_ms,
                    "iso_time": iso_time,
                    "ph": row.get("ph"),
                    "o2": row.get("o2"),
                    "temp": row.get("temp"),
                }
            )

    with path.open("w", newline="", encoding="ascii") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row) -> None:
    with path.open("a", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def on_connect(client: mqtt.Client, _userdata, _flags, reason_code, properties=None) -> None:
    if reason_code != 0:
        logging.error("Failed to connect to broker: rc=%s", reason_code)
        return
    logging.info("Connected to %s:%s", BROKER, PORT)
    for topic in TOPICS:
        client.subscribe(topic, qos=1)
        logging.info("Subscribed to %s", topic)


def on_message(client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
    topic = message.topic
    key = topic_map.get(topic)
    if key is None:
        logging.warning("Received message on unexpected topic: %s", topic)
        return

    try:
        buffer[key] = float(message.payload.decode().strip())
    except ValueError:
        logging.warning("Non-numeric payload on %s: %r", topic, message.payload)
        return

    if all(value is not None for value in buffer.values()):
        timestamp_ms = int(time.time() * 1000)
        iso_time = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()
        append_row(
            OUT_CSV,
            [timestamp_ms, iso_time, buffer["ph"], buffer["o2"], buffer["temp"]],
        )
        logging.debug("Logged row: %s", buffer)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ensure_csv_header(OUT_CSV)

    client = mqtt.Client(
        client_id="aq_logger",
        clean_session=True,
        protocol=mqtt.MQTTv311,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, KEEPALIVE)
    logging.info("Starting MQTT loop")
    client.loop_forever()


if __name__ == "__main__":
    main()
