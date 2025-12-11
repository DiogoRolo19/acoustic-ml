import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import AudioFeatureDataset
import pandas as pd
import numpy as np
from utils.model import MarineClassifier

SPECIES_COLUMN = "simple_species"

class AudioTester:
    def __init__(self, model_path="cnn14_finetuned.pth", csv_path="dataset_splits.csv", batch_size=16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CSV
        self.df = pd.read_csv(csv_path)
        self.df[SPECIES_COLUMN] = self.df["species"].apply(lambda x: x.split()[-1])
        self.species_list = sorted(self.df[SPECIES_COLUMN].unique())
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        self.num_classes = len(self.species_list)

        # Prepare test dataset
        test_df = self.df[self.df["split"] == "test"]
        self.test_ds = AudioFeatureDataset(test_df, self.species_to_idx)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, num_workers=4)

        # Load model
        self.classifier = MarineClassifier(self.num_classes, self.device)
        self.classifier.load_finetuned(model_path)
        self.classifier.model.eval()

    def evaluate(self):
        criterion = nn.CrossEntropyLoss()
        avg_loss, accuracy = self.classifier.evaluate(self.test_loader, criterion)
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy


if __name__ == "__main__":
    from utils.pipeline import run_full_pipeline
    
    # Run pipeline steps if needed
    if run_full_pipeline():
        # Test the model
        tester = AudioTester(model_path="cnn14_finetuned.pth")
        tester.evaluate()
