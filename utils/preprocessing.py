import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
INPUT_ROOT = "dataset"
OUTPUT_ROOT = "preprocessed"
TARGET_SR = 44100
MIN_DURATION = 0.5
MAX_DURATION = 300
CHUNK_DURATION = 10
MIN_CHUNK_SAMPLES = 2048

READABLE = False  # True = save as WAV, False = save as NPY

def process_file(file_path, output_dir):
    try:
        # Load audio and resample
        y, sr = librosa.load(file_path, sr=TARGET_SR)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        duration = librosa.get_duration(y=y, sr=sr)

        if duration < MIN_DURATION:
            print(f"Skipping {file_path}: too short ({duration:.2f}s)")
            return
        if duration > MAX_DURATION:
            print(f"Skipping {file_path}: too long ({duration:.2f}s)")
            return

        # Normalize amplitude
        y = y / max(abs(y))

        # Chunk size in samples
        chunk_size = int(CHUNK_DURATION * sr)

        # Number of chunks
        n_chunks = int(np.ceil(len(y) / chunk_size))

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size

            chunk = y[start:end]

            # PAD the chunk if it's shorter than chunk_size
            if len(chunk) < chunk_size:
                padding = np.zeros(chunk_size - len(chunk), dtype=chunk.dtype)
                chunk = np.concatenate([chunk, padding])

            # Save output
            if READABLE:
                out_file = os.path.join(output_dir, f"{base_name}_chunk{i}.wav")
                sf.write(out_file, chunk, sr)
            else:
                out_file = os.path.join(output_dir, f"{base_name}_chunk{i}.npy")
                np.save(out_file, chunk)

        print(f"Processed {file_path} ({duration:.2f}s)")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def preprocess_dataset(input_root=INPUT_ROOT, output_root=OUTPUT_ROOT):
    Path(output_root).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, rel_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for file in files:
            if file.lower().endswith((".wav")):
                input_file = os.path.join(root, file)
                process_file(input_file, output_dir)

if __name__ == "__main__":
    print("Starting preprocessing...")
    preprocess_dataset()
    print("Preprocessing finished!")
