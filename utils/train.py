import os
import numpy as np
import pandas as pd
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import AudioFeatureDataset
from utils.config import SPECIES_COLUMN,SAVE_INTERVAL,EPOCHS
from utils.model import MarineClassifier


import re

def get_epoch_from_filename(path):
    m = re.search(r"epoch_(\d+)", path)
    return int(m.group(1)) if m else float('inf')


def save_model_rolling(model, epoch, final_path="cnn14_finetuned.pth", temp_dir="temp_finetuning", max_temp_files=2):
    """
    Rolling save logic:
    - If final_path exists, move it to temp_dir with its epoch number.
    - Keep only the last max_temp_files in temp_dir.
    - Save current model as final_path.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Move current final to temp dir if it exists
    if os.path.exists(final_path):
        temp_filename = os.path.join(temp_dir, f"cnn14_epoch_{epoch}.pth")
        shutil.move(final_path, temp_filename)
        print(f"Moved previous final model to temporary: {temp_filename}")

        # Cleanup old temp files
        temp_files = sorted(
            [f for f in os.listdir(temp_dir) if f.endswith(".pth")],
            key=get_epoch_from_filename
        )

        while len(temp_files) > max_temp_files:
            old_file = os.path.join(temp_dir, temp_files.pop(0))
            os.remove(old_file)
            print(f"Deleted old temp model: {old_file}")

    # Save the current model as final
    model.save(final_path)
    print(f"Saved current model as final: {final_path}")

def main():

    # --------------------------
    # Load CSV
    # --------------------------
    df = pd.read_csv("dataset_splits.csv")

    # Keep only the last word in each class name
    df[SPECIES_COLUMN] = df["species"].apply(lambda x: x.split()[-1])

    species_list = sorted(df[SPECIES_COLUMN].unique())
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)

    print("Found classes:", species_to_idx)

    # --------------------------
    # Dataset splits
    # --------------------------
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]

    train_ds = AudioFeatureDataset(train_df, species_to_idx)
    val_ds   = AudioFeatureDataset(val_df, species_to_idx)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=16, num_workers=4)

    # --------------------------
    # Initialize classifier and train
    # --------------------------
    classifier = MarineClassifier(num_classes)
    classifier.load_pretrained()

    # Count samples per class
    counts = train_df[SPECIES_COLUMN].value_counts()
    counts_ordered = np.array([counts.get(species, 0) for species in species_list], dtype=np.float32)

    class_weights = 1.0 / counts_ordered

    class_weights = class_weights / class_weights.sum()

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, classifier.model.parameters()),
        lr=1e-4
    )

    
    for epoch in range(EPOCHS):
        train_loss, train_acc = classifier.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = classifier.evaluate(val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        # Save rolling every x epochs
        if (epoch+1) % SAVE_INTERVAL == 0 or epoch+1 == EPOCHS:
            save_model_rolling(classifier, epoch)
    print("Training complete!")


if __name__ == "__main__":
    main()
