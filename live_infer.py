import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow import keras

BROKER = "jetsion.com"
PORT = 1883
KEEPALIVE = 60

IN_TOPICS = [
    "jetsion/30AEA4903C20/WaterPH",
    "jetsion/30AEA4903C20/WaterO2",
    "jetsion/30AEA4903C20/WaterTemp",
]
OUT_TOPIC = "jetsion/30AEA4903C20/WaterQuality/Status"

META_PATH = Path("aq_meta.pkl")
MODEL_PRIORITIES = [Path("aq_dnn.keras"), Path("aq_dnn.h5")]

LEGEND = {
    "pH_low": "pH Low (酸鹼值過低)",
    "pH_high": "pH High (酸鹼值過高)",
    "DO_low": "Dissolved Oxygen Low (溶氧不足)",
    "Temp_high": "Temperature High (水溫過高)",
    "normal": "Within Thresholds (各項數值正常)",
}

TOPIC_SUFFIX_MAP = {
    "WaterPH": "ph",
    "WaterO2": "o2",
    "WaterTemp": "temp",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class LiveInference:
    def __init__(self) -> None:
        if not META_PATH.exists():
            raise FileNotFoundError("Metadata not found. Run train_classifier.py first.")

        meta = joblib.load(META_PATH)
        self.scaler = meta["scaler"]
        self.classes = meta["classes"]

        self.model_path = self._resolve_model_path()
        self.model = keras.models.load_model(self.model_path)
        self.buffer = {"ph": None, "o2": None, "temp": None}

    @staticmethod
    def _resolve_model_path() -> Path:
        for path in MODEL_PRIORITIES:
            if path.exists():
                return path
        raise FileNotFoundError("Model file not found. Run train_classifier.py first.")

    def on_message(self, client: mqtt.Client, _userdata, message: mqtt.MQTTMessage) -> None:
        topic = message.topic
        payload = message.payload.decode().strip()

        suffix = topic.split("/")[-1]
        key = TOPIC_SUFFIX_MAP.get(suffix)
        if key is None:
            logging.debug("Ignoring unknown topic %s", topic)
            return

        try:
            self.buffer[key] = float(payload)
        except ValueError:
            logging.warning("Invalid payload %r on %s", payload, topic)
            return

        if all(value is not None for value in self.buffer.values()):
            features = self.scaler.transform(
                [[self.buffer["ph"], self.buffer["o2"], self.buffer["temp"]]]
            )
            probabilities = self.model.predict(features, verbose=0)[0]
            best_idx = int(np.argmax(probabilities))
            best_label = self.classes[best_idx]
            confidence = float(probabilities[best_idx])
            label_text = LEGEND.get(best_label, best_label)
            payload_out = f"{best_label}:{confidence:.3f}:{label_text}"
            client.publish(OUT_TOPIC, payload_out, qos=1, retain=True)
            logging.info("Published %s", payload_out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Loading model and scaler")
    live = LiveInference()

    client = mqtt.Client(
        client_id="aq_infer",
        clean_session=True,
        protocol=mqtt.MQTTv311,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_message = live.on_message
    client.connect(BROKER, PORT, KEEPALIVE)

    for topic in IN_TOPICS:
        client.subscribe(topic, qos=1)
        logging.info("Subscribed to %s", topic)

    logging.info("Entering MQTT loop (model: %s)", live.model_path)
    client.loop_forever()


if __name__ == "__main__":
    main()
