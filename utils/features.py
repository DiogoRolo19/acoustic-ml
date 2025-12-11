import os
import numpy as np
import librosa
from utils.config import SR,PREPROCESSED_ROOT,FEATURES_ROOT,MEL_BINS,N_MFCC

# --- CONFIGURATION ---
INPUT_ROOT = PREPROCESSED_ROOT
OUTPUT_ROOT = FEATURES_ROOT

def extract_spectrogram(wav):
    return np.abs(librosa.stft(wav))

def extract_mfcc(wav, sr=SR, n_mfcc=N_MFCC):
    return librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc)

def extract_logmel(wav, sr=SR, n_mels=MEL_BINS):
    spec = librosa.stft(wav) # Compute STFT
    mel = librosa.feature.melspectrogram(S=np.abs(spec)**2, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel)

def extract_zcr(wav):
    return librosa.feature.zero_crossing_rate(wav)

def extract_spectral_centroid(wav, sr=SR):
    return librosa.feature.spectral_centroid(y=wav, sr=sr)

def extract_spectral_bandwidth(wav, sr=SR):
    return librosa.feature.spectral_bandwidth(y=wav, sr=sr)

def extract_spectral_rolloff(wav, sr=SR):
    return librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.85)

def extract_rms(wav):
    return librosa.feature.rms(y=wav)

def extract_deltas(mfcc):
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return delta, delta2


def extract_all_features(wav, sr=SR, n_mfcc=N_MFCC):
    spec = extract_spectrogram(wav)
    mfcc = extract_mfcc(wav, sr, n_mfcc)
    logmel = extract_logmel(wav, sr, MEL_BINS)
    zcr = extract_zcr(wav)
    centroid = extract_spectral_centroid(wav, sr)
    bandwidth = extract_spectral_bandwidth(wav, sr)
    rolloff = extract_spectral_rolloff(wav, sr)
    rms = extract_rms(wav)
    delta, delta2 = extract_deltas(mfcc)

    return {
        "spectrogram": spec,
        "mfcc": mfcc,
        "logmel": logmel,
        "zcr": zcr,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "rms": rms,
        "delta": delta,
        "delta2": delta2
    }


def save_features(save_path, features):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **features)


def process_preprocessed_dir(base_dir=INPUT_ROOT, out_dir=OUTPUT_ROOT,
                             sr=SR, n_mfcc=N_MFCC):

    for species in os.listdir(base_dir):
        species_dir = os.path.join(base_dir, species)
        if not os.path.isdir(species_dir):
            continue

        for file in os.listdir(species_dir):
            if not file.endswith(".npy"):
                continue

            wav_path = os.path.join(species_dir, file)
            wav = np.load(wav_path)

            features = extract_all_features(wav, sr, n_mfcc)

            save_name = file.replace(".npy", ".npz")
            save_path = os.path.join(out_dir, species, save_name)

            save_features(save_path, features)

            print(f"Saved features → {save_path}")


if __name__ == "__main__":
    process_preprocessed_dir()
    print("\nAll features extracted and saved!")
