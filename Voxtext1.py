import json
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import sounddevice as sd
from vosk import Model, KaldiRecognizer


# -------------------------
# Config
# -------------------------
MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

PREFERRED_MIC_SUBSTRING = "Adafruit"   # set None to skip auto-pick
DEVICE = None                         # set to int to force device index

MIN_FINAL_CHARS = 2


def pick_input_device(preferred_substring: str | None):
    if not preferred_substring:
        return None
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            name = d.get("name", "")
            if preferred_substring.lower() in name.lower():
                return i
    return None


def normalize_text(t: str) -> str:
    return " ".join((t or "").lower().strip().split())


def should_print(new_text: str, last_text: str) -> bool:
    n = normalize_text(new_text)
    l = normalize_text(last_text)
    if not n:
        return False
    if n == l:
        return False
    # suppress near-duplicates: if one contains the other, keep longer
    if l and (n in l or l in n):
        return len(n) > len(l)
    return True


class PushToTalkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VoxText - Push to Talk (Offline Vosk)")

        # UI
        self.status_var = tk.StringVar(value="Idle (not listening)")
        self.btn_var = tk.StringVar(value="Start Listening")

        self.status_label = ttk.Label(root, textvariable=self.status_var)
        self.status_label.pack(padx=12, pady=(12, 6), anchor="w")

        self.toggle_btn = ttk.Button(root, textvariable=self.btn_var, command=self.toggle_listening)
        self.toggle_btn.pack(padx=12, pady=6, fill="x")

        self.output = tk.Text(root, height=12, wrap="word")
        self.output.pack(padx=12, pady=(6, 12), fill="both", expand=True)
        self.output.insert("end", "Click Start Listening → talk → click Stop Listening to finalize.\n")

        # Audio/recognizer state
        self.listening = False
        self.stream = None
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=200)

        self.model = None
        self.rec = None
        self.last_final = ""

        # Worker thread control
        self.worker_stop = threading.Event()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)

        self.init_vosk()
        self.worker_thread.start()

        # Clean shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_vosk(self):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_DIR}")

        self.model = Model(str(MODEL_DIR))
        self.rec = KaldiRecognizer(self.model, SAMPLE_RATE)

        # CPU reducers
        try:
            self.rec.SetWords(False)
        except Exception:
            pass
        try:
            self.rec.SetMaxAlternatives(0)
        except Exception:
            pass

    def audio_callback(self, indata, frames, time_info, status):
        # Only queue audio while listening; otherwise discard
        if not self.listening:
            return
        try:
            self.audio_q.put_nowait(bytes(indata))
        except queue.Full:
            pass  # drop to keep latency stable

    def start_stream(self):
        dev = DEVICE
        if dev is None:
            dev = pick_input_device(PREFERRED_MIC_SUBSTRING)

        # You can print device name into UI for confidence
        try:
            if dev is not None:
                d = sd.query_devices(dev)
                self.append_line(f"Using input device {dev}: {d['name']}")
            else:
                self.append_line("Using default input device")
        except Exception:
            pass

        # Raw int16 stream = lower CPU and direct Vosk format
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=dev,
            channels=1,
            dtype="int16",
            callback=self.audio_callback,
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def toggle_listening(self):
        if not self.listening:
            # Start listening
            self.listening = True
            self.status_var.set("Listening... (click Stop when done)")
            self.btn_var.set("Stop Listening")
            self.append_line("▶ Listening started")
            if self.stream is None:
                self.start_stream()
            # Reset recognizer for a fresh utterance
            self.rec.Reset()
        else:
            # Stop listening + finalize
            self.listening = False
            self.status_var.set("Finalizing...")
            self.btn_var.set("Start Listening")
            self.append_line("■ Listening stopped (finalizing...)")

            # Drain any remaining queued audio quickly
            time_limit = time.time() + 0.4
            while time.time() < time_limit:
                try:
                    b = self.audio_q.get_nowait()
                except queue.Empty:
                    break
                self.rec.AcceptWaveform(b)

            # Finalize result
            try:
                res = json.loads(self.rec.FinalResult())
                text = (res.get("text") or "").strip()
            except Exception:
                text = ""

            if len(text) >= MIN_FINAL_CHARS and should_print(text, self.last_final):
                self.last_final = text
                self.append_line(f"FINAL: {text}")
            else:
                self.append_line("FINAL: (nothing detected)")

            self.status_var.set("Idle (not listening)")

    def worker_loop(self):
        """
        Background worker: while listening, feed audio to recognizer.
        We do NOT print partials. We just keep recognizer updated.
        """
        while not self.worker_stop.is_set():
            try:
                b = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # If user clicked stop, ignore queued audio
            if not self.listening:
                continue

            try:
                self.rec.AcceptWaveform(b)
            except Exception:
                # Avoid crashing the UI if something weird happens
                pass

    def append_line(self, line: str):
        self.output.insert("end", line + "\n")
        self.output.see("end")

    def on_close(self):
        self.worker_stop.set()
        self.listening = False
        self.stop_stream()
        self.root.destroy()


def main():
    root = tk.Tk()
    # nicer default styling on Pi
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = PushToTalkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()