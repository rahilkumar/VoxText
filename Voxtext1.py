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


def pick_input_device(preferred_substring: str = "Adafruit"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and preferred_substring.lower() in d.get("name", "").lower():
            return i
    return None


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Voxtext - Vosk STT (Push Button)")

        self.status_var = tk.StringVar(value="Stopped")
        self.btn_var = tk.StringVar(value="Start Listening")

        ttk.Label(root, textvariable=self.status_var).pack(padx=12, pady=(12, 6), anchor="w")
        ttk.Button(root, textvariable=self.btn_var, command=self.toggle).pack(padx=12, pady=6, fill="x")

        self.out = tk.Text(root, height=14, wrap="word")
        self.out.pack(padx=12, pady=(6, 12), fill="both", expand=True)
        self.out.insert("end", "Click Start Listening → talk → Stop Listening to finalize.\n\n")

        self.listening = False
        self.stop_event = threading.Event()

        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)

        self.model = None
        self.rec = None
        self.stream = None
        self.worker = threading.Thread(target=self.stt_loop, daemon=True)

        self.last_final_text = ""
        self.partial_last = ""
        self.last_partial_print_t = 0.0
        self.PARTIAL_THROTTLE_S = 0.08

        self.init_vosk_and_stream()
        self.worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def append(self, msg: str):
        self.out.insert("end", msg)
        self.out.see("end")

    def init_vosk_and_stream(self):
        if not MODEL_DIR.exists():
            self.append(f"Model not found: {MODEL_DIR}\n")
            self.append("Put the model folder under models/ and update MODEL_DIR.\n")
            raise SystemExit(1)

        dev = DEVICE if DEVICE is not None else pick_input_device("Adafruit")

        if dev is not None:
            d = sd.query_devices(dev)
            self.append(f"Using input device {dev}: {d['name']}\n")
        else:
            self.append("Using default input device\n")

        self.model = Model(str(MODEL_DIR))
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.rec.SetWords(True)

        # Start stream; callback only queues when listening=True
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                device=dev,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self.append(f"\nERROR starting audio stream:\n{e}\n")
            self.append("Fix: check mic is plugged in, correct device, and PortAudio.\n")
            raise

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            # don't spam UI; just stderr
            print(status, file=sys.stderr)

        # Push-to-talk: only queue audio while listening
        if not self.listening:
            return

        pcm16 = (indata[:, 0] * 32767).astype(np.int16)
        try:
            self.audio_q.put_nowait(pcm16)
        except queue.Full:
            pass

    def toggle(self):
        if not self.listening:
            # Start
            self.listening = True
            self.status_var.set("Listening...")
            self.btn_var.set("Stop Listening")

            self.last_final_text = ""
            self.partial_last = ""
            self.last_partial_print_t = 0.0
            self.rec.Reset()

            # Clear any old queued chunks (fresh start)
            while True:
                try:
                    self.audio_q.get_nowait()
                except queue.Empty:
                    break

            self.append("▶ Listening started\n")
        else:
            # Stop + finalize
            self.listening = False
            self.status_var.set("Finalizing...")
            self.btn_var.set("Start Listening")
            self.append("■ Listening stopped (finalizing...)\n")

            # Flush whatever is left in queue quickly
            deadline = time.time() + 0.4
            while time.time() < deadline:
                try:
                    pcm16 = self.audio_q.get_nowait()
                except queue.Empty:
                    break
                self.rec.AcceptWaveform(pcm16.tobytes())

            # Finalize
            try:
                res = json.loads(self.rec.FinalResult())
                text = (res.get("text") or "").strip()
            except Exception:
                text = ""

            if len(text) >= MIN_FINAL_CHARS and text != self.last_final_text:
                self.last_final_text = text
                self.append(f"FINAL: {text}\n\n")
            else:
                self.append("FINAL: (nothing detected)\n\n")

            self.status_var.set("Stopped")

    def stt_loop(self):
        while not self.stop_event.is_set():
            try:
                pcm16 = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.listening:
                continue

            data_bytes = pcm16.tobytes()

            if self.rec.AcceptWaveform(data_bytes):
                res = json.loads(self.rec.Result())
                text = (res.get("text") or "").strip()

                if len(text) >= MIN_FINAL_CHARS and text != self.last_final_text:
                    self.last_final_text = text
                    self.root.after(0, lambda t=text: self.append(f"FINAL: {t}\n\n"))
            else:
                # Optional: show partial in the status line (not in textbox)
                pres = json.loads(self.rec.PartialResult())
                p = (pres.get("partial") or "").strip()
                now = time.time()
                if p and p != self.partial_last and (now - self.last_partial_print_t) >= self.PARTIAL_THROTTLE_S:
                    self.partial_last = p
                    self.last_partial_print_t = now
                    self.root.after(0, lambda t=p: self.status_var.set(f"Listening... {t}"))

    def on_close(self):
        self.stop_event.set()
        self.listening = False
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