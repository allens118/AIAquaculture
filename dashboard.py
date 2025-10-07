import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import paho.mqtt.client as mqtt
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

BROKER = "jetsion.com"
PORT = 1883
KEEPALIVE = 60

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

IN_TOPICS = [
    "jetsion/30AEA4903C20/WaterPH",
    "jetsion/30AEA4903C20/WaterO2",
    "jetsion/30AEA4903C20/WaterTemp",
]
STATUS_TOPIC = "jetsion/30AEA4903C20/WaterQuality/Status"

SENSOR_KEYS = {
    "WaterPH": "ph",
    "WaterO2": "o2",
    "WaterTemp": "temp",
}

PH_LOW, PH_HIGH = 7.1, 8.5
DO_LOW = 7.64
TEMP_HIGH = 26.3

SENSOR_NAMES = {
    "ph": "pH 值 (pH)",
    "o2": "溶氧 (Dissolved Oxygen)",
    "temp": "水溫 (Water Temperature)",
}


class ManagedProcess:
    def __init__(self, name: str, script: str, append_log, state_callback: Optional[Callable[[], None]] = None):
        self.name = name
        self.script = script
        self.append_log = append_log
        self.state_callback = state_callback
        self.process: Optional[subprocess.Popen] = None
        self.output_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        proc = self.process
        return proc is not None and proc.poll() is None

    def start(self) -> bool:
        with self.lock:
            if self.is_running():
                self.append_log(self.name, "程序已在執行中 (Process already running)")
                return False
            cmd = [PYTHON, str(SCRIPT_DIR / self.script)]
            self.append_log(self.name, f"啟動指令 (Launch command): {' '.join(cmd)}")
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except Exception as exc:
                self.append_log(self.name, f"啟動失敗 (Failed to launch): {exc}")
                self.process = None
                self.output_thread = None
                self._notify_state_change()
                return False

            self.output_thread = threading.Thread(target=self._forward_output, daemon=True)
            self.output_thread.start()
            self._notify_state_change()
            return True

    def _forward_output(self) -> None:
        proc = self.process
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            self.append_log(self.name, line.rstrip())
        code = proc.poll()
        self.append_log(self.name, f"程序結束，返回碼 (Process exited, code) {code}")
        with self.lock:
            self.process = None
            self.output_thread = None
        self._notify_state_change()

    def stop(self) -> bool:
        with self.lock:
            if not self.is_running():
                self.append_log(self.name, "程序未在執行 (Process not running)")
                return False
            self.append_log(self.name, "正在嘗試停止程序 (Attempting graceful stop)...")
            assert self.process is not None
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.append_log(self.name, "強制結束程序 (Force killing process)")
                self.process.kill()
            finally:
                self.process = None
                self.output_thread = None
        self._notify_state_change()
        return True

    def _notify_state_change(self) -> None:
        if self.state_callback:
            try:
                self.state_callback()
            except Exception:
                pass


class Dashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("水產養殖監測控制台 (Aquaculture Monitoring Dashboard)")
        self.root.configure(bg="#eef2f7")
        self.root.option_add("*Font", "{Segoe UI} 10")

        if not getattr(sys, "base_prefix", sys.prefix) == sys.prefix:
            tk.messagebox.showwarning(
                "虛擬環境 (Virtual Environment)",
                "偵測到目前未使用虛擬環境執行，部分套件可能不相容。\n"
                "請先執行 .\\.venv\\Scripts\\Activate.ps1 後再開啟此介面。",
            )

        self.sensor_values = {"ph": None, "o2": None, "temp": None}
        self.status_text = tk.StringVar(value="尚未收到推論結果 (No inference yet)")
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.logger_proc = ManagedProcess(
            "資料蒐集 (Data Logging)",
            "mqtt_logger.py",
            self.enqueue_log,
            self._process_state_changed,
        )
        self.infer_proc = ManagedProcess(
            "即時推論 (Live Inference)",
            "live_infer.py",
            self.enqueue_log,
            self._process_state_changed,
        )
        self.train_thread: Optional[threading.Thread] = None
        self.training = False

        self._build_styles()
        self._build_ui()
        self._setup_mqtt()
        self._update_sensor_labels()
        self._drain_log_queue()
        self._update_control_states()

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=0, relief="flat")
        style.configure("Card.TLabelframe.Label", background="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("Card.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        style.map("Card.TButton", background=[("active", "#3f74e3")], foreground=[("active", "#ffffff")])
        style.configure("Value.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 13, "bold"))

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=16, style="Card.TLabelframe")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        sensor_frame = ttk.LabelFrame(
            main_frame,
            text="即時感測數據 (Real-time Sensor Readings)",
            style="Card.TLabelframe",
            padding=12,
        )
        sensor_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        sensor_frame.columnconfigure((0, 1, 2), weight=1)

        self.ph_var = tk.StringVar(value=f"{SENSOR_NAMES['ph']}: --")
        self.o2_var = tk.StringVar(value=f"{SENSOR_NAMES['o2']}: --")
        self.temp_var = tk.StringVar(value=f"{SENSOR_NAMES['temp']}: --")

        ttk.Label(sensor_frame, textvariable=self.ph_var, style="Value.TLabel").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Label(sensor_frame, textvariable=self.o2_var, style="Value.TLabel").grid(
            row=0, column=1, sticky="w", padx=6, pady=4
        )
        ttk.Label(sensor_frame, textvariable=self.temp_var, style="Value.TLabel").grid(
            row=0, column=2, sticky="w", padx=6, pady=4
        )

        thresh_frame = ttk.LabelFrame(
            main_frame,
            text="異常門檻 (Alert Thresholds)",
            style="Card.TLabelframe",
            padding=12,
        )
        thresh_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        thresh_frame.columnconfigure((0, 1, 2), weight=1)

        ttk.Label(
            thresh_frame,
            text=f"pH 異常 (pH alert): < {PH_LOW} 或 (or) > {PH_HIGH}",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Label(
            thresh_frame,
            text=f"溶氧過低 (Low dissolved oxygen): < {DO_LOW}",
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)
        ttk.Label(
            thresh_frame,
            text=f"水溫過高 (High water temperature): > {TEMP_HIGH}",
        ).grid(row=0, column=2, sticky="w", padx=6, pady=2)

        status_frame = ttk.LabelFrame(
            main_frame,
            text="AI 推論結果 (AI Inference Result)",
            style="Card.TLabelframe",
            padding=12,
        )
        status_frame.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(status_frame, textvariable=self.status_text, style="Status.TLabel").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )

        button_frame = ttk.LabelFrame(
            main_frame,
            text="作業控制 (Operations)",
            style="Card.TLabelframe",
            padding=12,
        )
        button_frame.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        button_frame.columnconfigure((0, 1, 2), weight=1)

        self.btn_logger = ttk.Button(
            button_frame,
            text="啟動資料蒐集 (Start Logging)",
            style="Card.TButton",
            command=self.start_logger,
        )
        self.btn_logger.grid(row=0, column=0, sticky="ew", padx=6, pady=4)

        self.btn_logger_stop = ttk.Button(
            button_frame,
            text="停止資料蒐集 (Stop Logging)",
            style="Card.TButton",
            command=self.stop_logger,
        )
        self.btn_logger_stop.grid(row=1, column=0, sticky="ew", padx=6, pady=4)

        self.btn_infer = ttk.Button(
            button_frame,
            text="啟動即時推論 (Start Inference)",
            style="Card.TButton",
            command=self.start_infer,
        )
        self.btn_infer.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        self.btn_infer_stop = ttk.Button(
            button_frame,
            text="停止即時推論 (Stop Inference)",
            style="Card.TButton",
            command=self.stop_infer,
        )
        self.btn_infer_stop.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        self.btn_train = ttk.Button(
            button_frame,
            text="訓練模型 (Train Model)",
            style="Card.TButton",
            command=self.run_training,
        )
        self.btn_train.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=6, pady=4)

        log_frame = ttk.LabelFrame(
            main_frame,
            text="日誌訊息 (Logs)",
            style="Card.TLabelframe",
            padding=12,
        )
        log_frame.grid(row=4, column=0, sticky="nsew")
        main_frame.rowconfigure(4, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, wrap="word")
        self.log_text.configure(font=("Consolas", 10), background="#1e1e24", foreground="#f5f5f5")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_mqtt(self):
        self.mqtt_client = mqtt.Client(client_id="aq_dashboard")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        try:
            self.mqtt_client.connect(BROKER, PORT, KEEPALIVE)
            self.mqtt_client.loop_start()
            self.enqueue_log("MQTT", "連線至 MQTT broker 中 (Connecting to MQTT broker)...")
        except Exception as exc:
            self.enqueue_log("MQTT", f"無法連線到 MQTT broker (Failed to connect): {exc}")
            messagebox.showerror("MQTT 連線失敗 (MQTT connection failed)", str(exc))

    def _on_mqtt_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            self.enqueue_log("MQTT", "連線成功，訂閱主題中 (Connected, subscribing topics)...")
            for topic in IN_TOPICS + [STATUS_TOPIC]:
                client.subscribe(topic, qos=1)
        else:
            self.enqueue_log("MQTT", f"連線失敗，返回碼 (Connection failed, code) {rc}")

    def _on_mqtt_message(self, _client, _userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode().strip()
        suffix = topic.split("/")[-1]
        if topic == STATUS_TOPIC:
            self.status_text.set(payload)
            return
        key = SENSOR_KEYS.get(suffix)
        if not key:
            return
        try:
            value = float(payload)
        except ValueError:
            self.enqueue_log("MQTT", f"收到非數值資料 (Non-numeric payload): {topic} -> {payload}")
            return
        self.sensor_values[key] = value

    def _update_sensor_labels(self):
        ph = self.sensor_values["ph"]
        o2 = self.sensor_values["o2"]
        temp = self.sensor_values["temp"]
        self.ph_var.set(
            f"{SENSOR_NAMES['ph']}: {ph:.2f}" if ph is not None else f"{SENSOR_NAMES['ph']}: --"
        )
        self.o2_var.set(
            f"{SENSOR_NAMES['o2']}: {o2:.2f}" if o2 is not None else f"{SENSOR_NAMES['o2']}: --"
        )
        self.temp_var.set(
            f"{SENSOR_NAMES['temp']}: {temp:.2f}" if temp is not None else f"{SENSOR_NAMES['temp']}: --"
        )
        self.root.after(500, self._update_sensor_labels)

    def enqueue_log(self, source: str, message: str):
        self.log_queue.put((source, message))

    def _drain_log_queue(self):
        try:
            while True:
                source, message = self.log_queue.get_nowait()
                self.log_text.insert("end", f"[{source}] {message}\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.root.after(200, self._drain_log_queue)

    def start_logger(self):
        if self.logger_proc.start():
            self._update_control_states()

    def stop_logger(self):
        if self.logger_proc.stop():
            self._update_control_states()

    def start_infer(self):
        if self.infer_proc.start():
            self._update_control_states()

    def stop_infer(self):
        if self.infer_proc.stop():
            self._update_control_states()

    def run_training(self):
        if self.training:
            messagebox.showinfo("訓練中 (Training)", "模型訓練尚未完成，請稍候 (Training in progress, please wait).")
            return
        self.training = True
        self._update_control_states()
        self.enqueue_log("訓練 (Training)", "開始模型訓練 (Start model training)...")

        def _train():
            cmd = [PYTHON, str(SCRIPT_DIR / "train_classifier.py")]
            proc = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                self.enqueue_log("訓練 (Training)", line.rstrip())
            code = proc.wait()
            self.enqueue_log("訓練 (Training)", f"訓練完成，返回碼 (Training finished, code) {code}")
            self.root.after(0, self._training_finished)

        self.train_thread = threading.Thread(target=_train, daemon=True)
        self.train_thread.start()

    def _training_finished(self):
        self.training = False
        self._update_control_states()

    def _process_state_changed(self):
        self.root.after(0, self._update_control_states)

    def _update_control_states(self):
        if self.logger_proc.is_running():
            self.btn_logger.configure(state=tk.DISABLED)
            self.btn_logger_stop.configure(state=tk.NORMAL)
        else:
            self.btn_logger.configure(state=tk.NORMAL)
            self.btn_logger_stop.configure(state=tk.DISABLED)

        if self.infer_proc.is_running():
            self.btn_infer.configure(state=tk.DISABLED)
            self.btn_infer_stop.configure(state=tk.NORMAL)
        else:
            self.btn_infer.configure(state=tk.NORMAL)
            self.btn_infer_stop.configure(state=tk.DISABLED)

        self.btn_train.configure(state=tk.DISABLED if self.training else tk.NORMAL)

    def on_close(self):
        if messagebox.askokcancel(
            "離開 (Exit)", "確定要關閉控制台並停止所有程序嗎？ (Close dashboard and stop all processes?)"
        ):
            self.logger_proc.stop()
            self.infer_proc.stop()
            if getattr(self, "mqtt_client", None):
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            self.root.destroy()


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
