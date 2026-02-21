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
DEVICE = None

# Output / filtering
MIN_FINAL_CHARS = 2
MIN_FINAL_WORDS = 1

# --- Auto noise gate (adaptive) ---
CALIBRATE_S = 1.0            # start-up calibration (stay quiet)
NOISE_ALPHA = 0.05           # how fast noise floor updates
MIN_THR = 120                # absolute minimum threshold (prevents too-low)
START_MULT = 2.0             # start speech when rms > noise*START_MULT
STOP_MULT  = 1.4             # stop speech when rms < noise*STOP_MULT (hysteresis)
MIN_SPEECH_MS = 60           # must be above threshold for this long to start
HANGOVER_MS = 300            # keep feeding a bit after it drops below stop threshold
FORCE_FINALIZE_SILENCE_MS = 350  # flush final after silence

# Duplicate protection
last_final_text = ""

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)

def pick_input_device(preferred_substring: str = "Adafruit"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and preferred_substring.lower() in d["name"].lower():
            return i
    return None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    pcm16 = (indata[:, 0] * 32767).astype(np.int16)
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        pass

def rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf)))

def normalize_text(t: str) -> str:
    return " ".join(t.lower().strip().split())

def is_new_final(new_text: str, old_text: str) -> bool:
    n = normalize_text(new_text)
    o = normalize_text(old_text)
    if not n:
        return False
    if n == o:
        return False
    # suppress near-duplicates: keep longer if one contains the other
    if o and (n in o or o in n):
        return len(n) > len(o)
    return True

def main():
    global last_final_text

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

    # CPU reducers (keep recognition quality)
    rec.SetWords(False)
    rec.SetMaxAlternatives(0)

    print("\n--- Voxtext: FINAL-only Streaming STT (Adaptive Noise Gate) ---")
    print(f"Calibrating for {CALIBRATE_S:.1f}s... stay quiet.\n")

    # Gate state
    in_speech = False
    above_ms = 0
    hang_ms = 0
    silence_ms = 0

    # Noise floor init
    noise_rms = None
    t_end = time.time() + CALIBRATE_S

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=dev,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        # Calibration phase
        while time.time() < t_end:
            pcm16 = audio_q.get()
            r = rms_int16(pcm16)
            noise_rms = r if noise_rms is None else (1 - NOISE_ALPHA) * noise_rms + NOISE_ALPHA * r

        if noise_rms is None:
            noise_rms = MIN_THR

        print("Listening. Ctrl+C to stop.\n")

        try:
            while True:
                pcm16 = audio_q.get()
                r = rms_int16(pcm16)

                start_thr = max(MIN_THR, noise_rms * START_MULT)
                stop_thr  = max(MIN_THR, noise_rms * STOP_MULT)

                # Update noise floor only when not in speech and below start threshold
                if (not in_speech) and (r < start_thr):
                    noise_rms = (1 - NOISE_ALPHA) * noise_rms + NOISE_ALPHA * r

                # Hysteresis gate
                if not in_speech:
                    if r >= start_thr:
                        above_ms += BLOCK_MS
                        if above_ms >= MIN_SPEECH_MS:
                            in_speech = True
                            hang_ms = HANGOVER_MS
                            silence_ms = 0
                    else:
                        above_ms = 0
                else:
                    if r < stop_thr:
                        hang_ms -= BLOCK_MS
                        silence_ms += BLOCK_MS
                        if hang_ms <= 0:
                            in_speech = False
                            above_ms = 0
                            hang_ms = 0
                    else:
                        hang_ms = HANGOVER_MS
                        silence_ms = 0

                data_bytes = pcm16.tobytes()

                # Feed recognizer only during speech
                if in_speech:
                    if rec.AcceptWaveform(data_bytes):
                        res = json.loads(rec.Result())
                        text = (res.get("text") or "").strip()
                        if len(text) >= MIN_FINAL_CHARS and len(text.split()) >= MIN_FINAL_WORDS:
                            if is_new_final(text, last_final_text):
                                last_final_text = text
                                print(text, flush=True)

                # Force finalize after brief silence (makes FINAL appear quickly)
                if (not in_speech) and silence_ms >= FORCE_FINALIZE_SILENCE_MS:
                    res = json.loads(rec.FinalResult())
                    text = (res.get("text") or "").strip()
                    if len(text) >= MIN_FINAL_CHARS and len(text.split()) >= MIN_FINAL_WORDS:
                        if is_new_final(text, last_final_text):
                            last_final_text = text
                            print(text, flush=True)
                    silence_ms = 0

        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == "__main__":
    main()