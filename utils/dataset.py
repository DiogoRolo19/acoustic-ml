import torch
from torch.utils.data import Dataset
import numpy as np

class AudioFeatureDataset(Dataset):
    def __init__(self, df, species_to_idx):
        self.df = df
        self.species_to_idx = species_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_path = row["preprocessed_path"]
        label = self.species_to_idx[row["simple_species"]]

        # Load numpy file
        x = np.load(npy_path)
        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return x, label