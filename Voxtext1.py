import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

DEVICE = None

# ---- Auto-gate settings (works across quiet → loud rooms) ----
CALIBRATE_S = 1.2          # listen before starting recognition
NOISE_ALPHA = 0.05         # noise floor smoothing (0.02–0.08)
MIN_THR = 180              # absolute floor (prevents too-low thresholds)

START_MULT = 2.2           # start speech if RMS > noise*START_MULT
STOP_MULT  = 1.6           # stop speech when RMS < noise*STOP_MULT (hysteresis)

MIN_SPEECH_MS = 60         # must be above threshold for this long to start
HANGOVER_MS = 350          # keep feeding after speech falls below stop threshold
FORCE_FINALIZE_SILENCE_MS = 450  # flush final if silence persists

# ---- Output filtering ----
MIN_FINAL_CHARS = 2
MIN_FINAL_WORDS = 1
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
        pass

def rms_int16_bytes(b: bytes) -> float:
    x = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(x * x)))

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
    rec.SetWords(False)
    rec.SetMaxAlternatives(0)

    print("\n--- Voxtext: FINAL-only Offline STT (Auto noise gate) ---")
    print(f"Calibrating for {CALIBRATE_S:.1f}s... stay quiet.\n")

    # Gate state
    in_speech = False
    above_ms = 0
    hang_ms = 0
    silence_ms = 0

    # Noise floor init from calibration
    noise_rms = None
    t_end = time.time() + CALIBRATE_S

    last_final_text = ""

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=dev,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        # Calibration phase
        while time.time() < t_end:
            b = audio_q.get()
            r = rms_int16_bytes(b)
            noise_rms = r if noise_rms is None else (1 - NOISE_ALPHA) * noise_rms + NOISE_ALPHA * r

        if noise_rms is None:
            noise_rms = MIN_THR

        print("Listening. Ctrl+C to stop.\n")

        try:
            while True:
                b = audio_q.get()
                r = rms_int16_bytes(b)

                # Update noise floor ONLY when not in speech AND below start threshold
                start_thr = max(MIN_THR, noise_rms * START_MULT)
                stop_thr  = max(MIN_THR, noise_rms * STOP_MULT)

                if (not in_speech) and (r < start_thr):
                    noise_rms = (1 - NOISE_ALPHA) * noise_rms + NOISE_ALPHA * r

                # Decide speech / silence using hysteresis thresholds
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
                    # in speech
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

                # Feed recognizer only during speech
                if in_speech:
                    if rec.AcceptWaveform(b):
                        res = json.loads(rec.Result())
                        text = (res.get("text") or "").strip()
                        if len(text) >= MIN_FINAL_CHARS and len(text.split()) >= MIN_FINAL_WORDS:
                            if is_new_final(text, last_final_text):
                                last_final_text = text
                                print(text, flush=True)

                # Force finalize after enough silence (makes FINAL appear quickly)
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