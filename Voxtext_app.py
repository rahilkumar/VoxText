import json
import queue
import sys
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import customtkinter as ctk

# -------------------------
# Config
# -------------------------
MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
BLOCK_MS = 30
DEVICE = None          # set to int index if you want to force
MIN_FINAL_CHARS = 1

# UI colors
BG = "#0b0f14"
CARD = "#111827"
TEXT = "#e5e7eb"
MUTED_RED = "#ef4444"
LIVE_GREEN = "#22c55e"
ACCENT = "#60a5fa"


def pick_input_device(preferred_substring: str = "Adafruit"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and preferred_substring.lower() in d.get("name", "").lower():
            return i
    return None


class VoxTextApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window
        self.title("VoxText")
        self.geometry("900x520")
        self.minsize(820, 480)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=BG)

        # State
        self.listening = False
        self.stop_event = threading.Event()
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=120)

        self.model = None
        self.rec = None
        self.stream = None

        self.last_final_text = ""
        self.partial_last = ""
        self.last_partial_t = 0.0
        self.PARTIAL_THROTTLE_S = 0.08

        # Build UI
        self._build_ui()

        # Init audio + worker
        self.init_vosk_and_stream()
        self.worker = threading.Thread(target=self.stt_loop, daemon=True)
        self.worker.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Top bar
        top = ctk.CTkFrame(self, fg_color=BG)
        top.pack(fill="x", padx=18, pady=(18, 10))

        title = ctk.CTkLabel(top, text="VoxText", text_color=TEXT,
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(side="left")

        self.status_var = ctk.StringVar(value="Muted")
        status = ctk.CTkLabel(top, textvariable=self.status_var, text_color="#9ca3af",
                              font=ctk.CTkFont(size=14))
        status.pack(side="right")

        # Main layout: left controls, right transcript
        main = ctk.CTkFrame(self, fg_color=BG)
        main.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        left = ctk.CTkFrame(main, fg_color=CARD, corner_radius=18)
        left.pack(side="left", fill="y", padx=(0, 12))

        right = ctk.CTkFrame(main, fg_color=CARD, corner_radius=18)
        right.pack(side="right", fill="both", expand=True)

        # Left panel content
        ctk.CTkLabel(left, text="Microphone", text_color=TEXT,
                     font=ctk.CTkFont(size=18, weight="bold")).pack(padx=18, pady=(18, 6), anchor="w")

        ctk.CTkLabel(left, text="Tap the circle to mute/unmute.\nGreen = listening.",
                     text_color="#9ca3af", justify="left").pack(padx=18, pady=(0, 14), anchor="w")

        # Big circular button (approx circle)
        self.mic_btn = ctk.CTkButton(
            left,
            text="",
            width=140,
            height=140,
            corner_radius=70,
            fg_color=MUTED_RED,
            hover_color="#dc2626",
            command=self.toggle_listening
        )
        self.mic_btn.pack(padx=24, pady=12)

        self.mic_label_var = ctk.StringVar(value="MUTED")
        self.mic_label = ctk.CTkLabel(left, textvariable=self.mic_label_var,
                                      text_color=TEXT, font=ctk.CTkFont(size=16, weight="bold"))
        self.mic_label.pack(pady=(6, 18))

        self.partial_var = ctk.StringVar(value="…")
        self.partial = ctk.CTkLabel(left, textvariable=self.partial_var, text_color=ACCENT,
                                    wraplength=250, justify="left")
        self.partial.pack(padx=18, pady=(0, 18), anchor="w")

        # Right panel: transcript
        ctk.CTkLabel(right, text="Transcript", text_color=TEXT,
                     font=ctk.CTkFont(size=18, weight="bold")).pack(padx=18, pady=(18, 10), anchor="w")

        self.textbox = ctk.CTkTextbox(right, fg_color="#0f172a", text_color=TEXT,
                                      corner_radius=14, border_width=1, border_color="#1f2937")
        self.textbox.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        self.textbox.insert("end", "Tap the red circle to start.\n\n")

    def append_final(self, msg: str):
        self.textbox.insert("end", msg)
        self.textbox.see("end")

    def init_vosk_and_stream(self):
        if not MODEL_DIR.exists():
            raise SystemExit(f"Model not found: {MODEL_DIR}")

        dev = DEVICE if DEVICE is not None else pick_input_device("Adafruit")

        if dev is not None:
            d = sd.query_devices(dev)
        else:
            d = sd.query_devices(sd.default.device[0])

        native_sr = int(d["default_samplerate"])
        block_size = int(native_sr * BLOCK_MS / 1000)

        # Recognizer sample rate should match incoming audio
        self.model = Model(str(MODEL_DIR))
        self.rec = KaldiRecognizer(self.model, native_sr)
        self.rec.SetWords(True)

        try:
            sd.check_input_settings(device=dev, channels=1, samplerate=native_sr, dtype="float32")
            self.stream = sd.InputStream(
                samplerate=native_sr,
                blocksize=block_size,
                device=dev,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            raise SystemExit(f"ERROR starting stream: {e}")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        if not self.listening:
            return

        pcm16 = (indata[:, 0] * 32767).astype(np.int16)
        try:
            self.audio_q.put_nowait(pcm16)
        except queue.Full:
            pass

    def toggle_listening(self):
        if not self.listening:
            # start
            self.listening = True
            self.mic_btn.configure(fg_color=LIVE_GREEN, hover_color="#16a34a")
            self.mic_label_var.set("LIVE")
            self.status_var.set("Listening…")
            self.partial_var.set("Speak now…")

            self.last_final_text = ""
            self.partial_last = ""
            self.last_partial_t = 0.0
            self.rec.Reset()

            while True:
                try:
                    self.audio_q.get_nowait()
                except queue.Empty:
                    break
        else:
            # stop + finalize
            self.listening = False
            self.mic_btn.configure(fg_color=MUTED_RED, hover_color="#dc2626")
            self.mic_label_var.set("MUTED")
            self.status_var.set("Muted")
            self.partial_var.set("…")

            # Drain a little then finalize
            deadline = time.time() + 0.4
            while time.time() < deadline:
                try:
                    pcm16 = self.audio_q.get_nowait()
                except queue.Empty:
                    break
                self.rec.AcceptWaveform(pcm16.tobytes())

            try:
                res = json.loads(self.rec.FinalResult())
                text = (res.get("text") or "").strip()
            except Exception:
                text = ""

            if len(text) >= MIN_FINAL_CHARS and text != self.last_final_text:
                self.last_final_text = text
                self.after(0, lambda t=text: self.append_final(f"FINAL: {t}\n\n"))
            else:
                self.after(0, lambda: self.append_final("FINAL: (nothing detected)\n\n"))

    def stt_loop(self):
        while not self.stop_event.is_set():
            try:
                pcm16 = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.listening:
                continue

            data = pcm16.tobytes()

            if self.rec.AcceptWaveform(data):
                res = json.loads(self.rec.Result())
                text = (res.get("text") or "").strip()
                if len(text) >= MIN_FINAL_CHARS and text != self.last_final_text:
                    self.last_final_text = text
                    self.after(0, lambda t=text: self.append_final(f"FINAL: {t}\n\n"))
            else:
                pres = json.loads(self.rec.PartialResult())
                p = (pres.get("partial") or "").strip()
                now = time.time()
                if p and p != self.partial_last and (now - self.last_partial_t) >= self.PARTIAL_THROTTLE_S:
                    self.partial_last = p
                    self.last_partial_t = now
                    self.after(0, lambda t=p: self.partial_var.set(t))

    def on_close(self):
        self.stop_event.set()
        self.listening = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = VoxTextApp()
    app.mainloop()