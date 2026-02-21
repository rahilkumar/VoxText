import json
import queue
import sys
import time
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

MODEL_DIR = Path(__file__).parent / "models" / "vosk-model-en-us-0.22-lgraph"
SAMPLE_RATE = 16000
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)
DEVICE = None

# Final-only behavior tuning
FORCE_FINALIZE_SILENCE_MS = 350   # lower = faster finals, try 250–450
MIN_FINAL_CHARS = 2

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
        pass  # drop audio to keep latency low

def norm(t: str) -> str:
    return " ".join(t.lower().strip().split())

def should_print(new_text: str, last_text: str) -> bool:
    n = norm(new_text)
    l = norm(last_text)
    if not n:
        return False
    if n == l:
        return False
    # If it’s basically the same phrase, keep only the longer/better one
    if l and (n in l or l in n):
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

    # CPU reducers
    rec.SetWords(False)
    rec.SetMaxAlternatives(0)

    print("\n--- Voxtext: FINAL-only (Reliable) ---")
    print("Speak. Pause briefly to get FINAL. Ctrl+C to stop.\n")

    last_final = ""
    silence_ms = 0

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
                b = audio_q.get()

                # Feed audio
                got_final = rec.AcceptWaveform(b)

                # Use partial internally ONLY to detect silence (don’t print it)
                # If partial becomes empty, you're likely in silence.
                p = json.loads(rec.PartialResult()).get("partial", "").strip()
                if p == "":
                    silence_ms += BLOCK_MS
                else:
                    silence_ms = 0

                # Normal final from Vosk endpointing
                if got_final:
                    text = json.loads(rec.Result()).get("text", "").strip()
                    if len(text) >= MIN_FINAL_CHARS and should_print(text, last_final):
                        last_final = text
                        print(text, flush=True)
                    continue

                # Force finalize after silence (faster FINAL)
                if silence_ms >= FORCE_FINALIZE_SILENCE_MS:
                    text = json.loads(rec.FinalResult()).get("text", "").strip()
                    if len(text) >= MIN_FINAL_CHARS and should_print(text, last_final):
                        last_final = text
                        print(text, flush=True)
                    silence_ms = 0

        except KeyboardInterrupt:
            print("\nStopping...")

if __name__ == "__main__":
    main()