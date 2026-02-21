import json
import queue
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -------------------------
# Config (tune these)
# -------------------------
MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"  # Update if you put your model somewhere else
SAMPLE_RATE = 16000          # Most Vosk models expect 16k
BLOCK_MS = 30                # 20–40ms is a good low-latency range
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)
DEVICE = None                # Set to an int device index to force a specific mic
MIN_FINAL_CHARS = 1          # Ignore empty finals

# Duplicate protection
last_final_text = ""

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

def pick_input_device(preferred_substring: str = "Adafruit"):
    """
    Optional helper: auto-pick a device that contains a substring in its name.
    Set preferred_substring=None to skip.
    """
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
        # Drop status messages to stderr so they don't pollute output
        print(status, file=sys.stderr)
    # indata is float32 [-1, 1]; convert to int16 PCM expected by Vosk
    pcm16 = (indata[:, 0] * 32767).astype(np.int16)
    try:
        audio_q.put_nowait(pcm16)
    except queue.Full:
        # If we're falling behind, drop audio to keep latency low
        pass

def main():
    global last_final_text

    if not MODEL_DIR.exists():
        print(f"Model not found: {MODEL_DIR}")
        print("Put a Vosk model folder inside Voxtext/models/ and update MODEL_DIR.")
        sys.exit(1)

    # Optional: auto-select your Adafruit USB mic if found
    dev = DEVICE
    if dev is None:
        dev = pick_input_device("Adafruit")

    if dev is not None:
        d = sd.query_devices(dev)
        print(f"Using input device {dev}: {d['name']}")
    else:
        print("Using default input device")

    model = Model(str(MODEL_DIR))
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    # For low-latency partials, keep this enabled:
    # (Vosk returns partials by default; the logic below prints them in-place.)

    print("\n--- Voxtext: Offline Streaming STT (Vosk) ---")
    print("Speak into the mic. Ctrl+C to stop.\n")

    partial_last = ""
    last_partial_print_t = 0.0
    PARTIAL_THROTTLE_S = 0.08   # reduce spam (prints at most ~12 times/sec)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        device=dev,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        try:
            while True:
                pcm16 = audio_q.get()
                data_bytes = pcm16.tobytes()

                if rec.AcceptWaveform(data_bytes):
                    # Final (utterance ended)
                    res = json.loads(rec.Result())
                    text = (res.get("text") or "").strip()

                    # Prevent blank finals and duplicates
                    if len(text) >= MIN_FINAL_CHARS and text != last_final_text:
                        last_final_text = text
                        partial_last = ""  # reset partial tracker
                        print(f"\nFINAL: {text}\n")
                else:
                    # Partial (still speaking)
                    pres = json.loads(rec.PartialResult())
                    p = (pres.get("partial") or "").strip()

                    now = time.time()
                    if p and p != partial_last and (now - last_partial_print_t) >= PARTIAL_THROTTLE_S:
                        partial_last = p
                        last_partial_print_t = now
                        # Print partial on same line (low-noise)
                        print(f"\rPARTIAL: {p}   ", end="", flush=True)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            # Don't call FinalResult() in continuous mode unless you want "flush"
            pass

if __name__ == "__main__":
    main()