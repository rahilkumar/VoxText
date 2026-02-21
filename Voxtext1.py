import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -------------------------
# Config
# -------------------------
MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000

BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

DEVICE = None  # set to int if you want

# ---- Noise / VAD gate tuning ----
# Start here for a loud demo place: 500-1200 is common.
RMS_THRESHOLD = 800

# Require a bit of "speech" before we start feeding Vosk (prevents random background hits)
MIN_SPEECH_MS = 120

# After speech ends, keep feeding for a bit so we don't cut off trailing words
HANGOVER_MS = 250

# How often to update noise floor (only during non-speech)
NOISE_UPDATE_ALPHA = 0.03  # lower = slower

# ---- Output filtering ----
MIN_FINAL_CHARS = 3
MIN_FINAL_WORDS = 2
SUPPRESS_SIMILAR = True

audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=120)

def pick_input_device(preferred_substring: str = "Adafruit"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and preferred_substring.lower() in d["name"].lower():
            return i
    return None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    try:
        audio_q.put_nowait(bytes(indata))
    except queue.Full:
        pass  # drop to keep latency low

def rms_int16(x: np.ndarray) -> float:
    # x: int16
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))

def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())

def is_new_final(new_text: str, last_text: str) -> bool:
    n = normalize_text(new_text)
    l = normalize_text(last_text)
    if not n:
        return False
    if n == l:
        return False
    if not SUPPRESS_SIMILAR or not l:
        return True
    # If one contains the other, keep the longer one (prevents repeat bursts)
    if n in l or l in n:
        return len(n) > len(l)
    return True

def main():
    if not MODEL_DIR.exists():
        print(f"Model not found: {MODEL_DIR}")
        sys.exit(1)

    dev = DEVICE if DEVICE is not None else pick_input_device("Adafruit")
    if dev is not None:
        d = sd.query_devices(dev)
        print(f"Using input device {dev}: {d['name']}")
    else:
        print("Using default input device")

    model = Model(str(MODEL_DIR))
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    # CPU reducers:
    rec.SetWords(False)          # big CPU saver (don’t need word timestamps)
    rec.SetMaxAlternatives(0)    # keep it simple

    print("\n--- Voxtext: FINAL-only Offline STT (Vosk) ---")
    print("Ctrl+C to stop.\n")

    # Gate state
    in_speech = False
    speech_ms = 0
    hang_ms = 0

    # Dynamic noise floor tracking (helps in changing environments)
    noise_rms = RMS_THRESHOLD * 0.6  # initial guess

    last_final_text = ""

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=dev,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        try:
            while True:
                data_bytes = audio_q.get()
                chunk = np.frombuffer(data_bytes, dtype=np.int16)

                r = rms_int16(chunk)

                # Update noise floor only when we're NOT in speech
                if not in_speech:
                    noise_rms = (1 - NOISE_UPDATE_ALPHA) * noise_rms + NOISE_UPDATE_ALPHA * r

                # Adaptive threshold: base threshold OR (noise floor * factor)
                # In loud places, noise floor rises; this helps ignore background.
                adaptive_thr = max(RMS_THRESHOLD, noise_rms * 2.2)

                is_loud = r >= adaptive_thr

                if is_loud:
                    speech_ms += BLOCK_MS
                    if not in_speech and speech_ms >= MIN_SPEECH_MS:
                        in_speech = True
                        hang_ms = HANGOVER_MS
                else:
                    if in_speech:
                        hang_ms -= BLOCK_MS
                        if hang_ms <= 0:
                            in_speech = False
                            speech_ms = 0
                            hang_ms = 0
                    else:
                        speech_ms = 0

                # Only feed recognizer when we consider it speech (or hangover)
                if in_speech:
                    if rec.AcceptWaveform(data_bytes):
                        res = json.loads(rec.Result())
                        text = (res.get("text") or "").strip()

                        # Basic quality gates to reduce garbage triggers
                        if len(text) >= MIN_FINAL_CHARS and len(text.split()) >= MIN_FINAL_WORDS:
                            if is_new_final(text, last_final_text):
                                last_final_text = text
                                print(text, flush=True)

        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == "__main__":
    main()