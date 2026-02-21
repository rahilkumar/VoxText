import json
import queue
import sys
import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -------------------------
# Config (same as yours)
# -------------------------
MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)
DEVICE = None
MIN_FINAL_CHARS = 1

# Duplicate protection
last_final_text = ""

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

def pick_input_device(preferred_substring: str = "Adafruit"):
    devices = sd.query_devices()
    candidates = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"]
            if preferred_substring and preferred_substring.lower() in name.lower():
                candidates.append((i, name))
    if candidates:
        return candidates[0][0]
    return None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    pcm16 = (indata[:, 0] * 32767).astype(np.int16)
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Voxtext - Vosk STT (Start/Stop)")

        self.status_var = tk.StringVar(value="Stopped")
        self.btn_var = tk.StringVar(value="Start Listening")

        ttk.Label(root, textvariable=self.status_var).pack(padx=12, pady=(12, 6), anchor="w")
        self.btn = ttk.Button(root, textvariable=self.btn_var, command=self.toggle)
        self.btn.pack(padx=12, pady=6, fill="x")

        self.out = tk.Text(root, height=14, wrap="word")
        self.out.pack(padx=12, pady=(6, 12), fill="both", expand=True)
        self.out.insert("end", "Click Start Listening → talk → Stop Listening to finalize.\n\n")

        # State
        self.listening = False
        self.stop_event = threading.Event()

        self.model = None
        self.rec = None

        self.partial_last = ""
        self.last_partial_print_t = 0.0
        self.PARTIAL_THROTTLE_S = 0.08

        self.stream = None
        self.worker = None

        self.init_vosk_and_stream()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_vosk_and_stream(self):
        if not MODEL_DIR.exists():
            self.append(f"Model not found: {MODEL_DIR}\n")
            self.append("Put the model folder under models/ and update MODEL_DIR.\n")
            raise SystemExit(1)

        dev = DEVICE
        if dev is None:
            dev = pick_input_device("Adafruit")

        if dev is not None:
            d = sd.query_devices(dev)
            self.append(f"Using input device {dev}: {d['name']}\n")
        else:
            self.append("Using default input device\n")

        self.model = Model(str(MODEL_DIR))
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.rec.SetWords(True)

        # Start audio stream immediately, but we'll ignore audio unless listening=True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=dev,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        )
        self.stream.start()

        # Start worker thread to process audio continuously
        self.worker = threading.Thread(target=self.stt_loop, daemon=True)
        self.worker.start()

    def append(self, msg: str):
        self.out.insert("end", msg)
        self.out.see("end")

    def toggle(self):
        global last_final_text

        if not self.listening:
            # Start
            self.listening = True
            self.status_var.set("Listening...")
            self.btn_var.set("Stop Listening")

            # Reset state so each press feels like a fresh utterance
            last_final_text = ""
            self.partial_last = ""
            self.last_partial_print_t = 0.0
            self.rec.Reset()

            self.append("▶ Listening started\n")
        else:
            # Stop and finalize
            self.listening = False
            self.status_var.set("Finalizing...")
            self.btn_var.set("Start Listening")

            # Flush any final words
            try:
                res = json.loads(self.rec.FinalResult())
                text = (res.get("text") or "").strip()
            except Exception:
                text = ""

            if len(text) >= MIN_FINAL_CHARS and text != last_final_text:
                last_final_text = text
                self.append(f"FINAL: {text}\n\n")
            else:
                self.append("FINAL: (nothing detected)\n\n")

            self.status_var.set("Stopped")
            self.append("■ Listening stopped\n")

    def stt_loop(self):
        global last_final_text

        while not self.stop_event.is_set():
            try:
                pcm16 = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.listening:
                continue

            data_bytes = pcm16.tobytes()

            if self.rec.AcceptWaveform(data_bytes):
                res = json.loads(self.rec.Result())
                text = (res.get("text") or "").strip()
                if len(text) >= MIN_FINAL_CHARS and text != last_final_text:
                    last_final_text = text
                    # UI updates must happen on main thread
                    self.root.after(0, lambda t=text: self.append(f"FINAL: {t}\n\n"))
            else:
                # show partial in UI (optional)
                pres = json.loads(self.rec.PartialResult())
                p = (pres.get("partial") or "").strip()

                now = time.time()
                if p and p != self.partial_last and (now - self.last_partial_print_t) >= self.PARTIAL_THROTTLE_S:
                    self.partial_last = p
                    self.last_partial_print_t = now
                    self.root.after(0, lambda t=p: self.status_var.set(f"Listening... {t}"))

    def on_close(self):
        self.stop_event.set()
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()