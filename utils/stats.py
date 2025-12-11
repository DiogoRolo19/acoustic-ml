import os
import librosa
import numpy as np
import pandas as pd


def dataset_stats(input_dir="dataset"):
    sample_rates = []
    durations = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                filepath = os.path.join(root, file)
                y, sr = librosa.load(filepath, sr=None, mono=True)
                sample_rates.append(sr)
                durations.append(len(y) / sr)

    sample_rates = np.array(sample_rates)
    durations = np.array(durations)

    print("===== Dataset Statistics =====")
    print(f"Number of files: {len(sample_rates)}")
    print(f"Sampling rate: min={sample_rates.min()}, max={sample_rates.max()}, mean={sample_rates.mean():.2f}")
    print(f"Duration (seconds): min={durations.min():.2f}, max={durations.max():.2f}, mean={durations.mean():.2f}")

def dataset_split_stats(csv_path="dataset_splits.csv"):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Extract the last word in the 'species' column
    df['species_last'] = df['species'].apply(lambda x: x.split()[-1])

    # Group by 'split' and 'species_last', then count rows
    grouped = df.groupby(['split', 'species_last']).size().reset_index(name='count')

    print("===== Dataset Split Statistics =====")
    for split in grouped['split'].unique():
        print(f"\nSplit: {split}")
        split_group = grouped[grouped['split'] == split]
        for _, row in split_group.iterrows():
            print(f"  Species: {row['species_last']:15} Count: {row['count']}")

if __name__ == "__main__":
    dataset_split_stats()