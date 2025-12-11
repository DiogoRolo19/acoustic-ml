import os
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import io

import matplotlib
matplotlib.use("Agg")  # use a non-GUI backend
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

from utils.dataset import AudioFeatureDataset
from utils.model import MarineClassifier

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # <-- allow all origins
    allow_credentials=True,
    allow_methods=["*"],          # <-- allow all HTTP methods
    allow_headers=["*"],          # <-- allow all headers
)

# Globals that will be loaded once at startup
df = None
test_df = None
species_list = None
species_to_idx = None
classifier = None

SPECIES_COLUMN = "simple_species"


@app.on_event("startup")
def startup_event():
    global df, test_df, species_list, species_to_idx, classifier
    
    print("Startup: loading CSV + model...")

    # Load CSV
    df = pd.read_csv("dataset_splits.csv")
    df[SPECIES_COLUMN] = df["species"].apply(lambda x: x.split()[-1])

    # Class mappings
    species_list = sorted(df[SPECIES_COLUMN].unique())
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    print("Found classes:", species_to_idx)

    # Test split only
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Load model once
    classifier = MarineClassifier(num_classes=len(species_list), device="cuda" if torch.cuda.is_available() else "cpu")
    classifier.load_finetuned("cnn14_finetuned.pth")
    classifier.model.eval()

    print("Startup complete: model + data loaded.")


@app.get("/samples")
def list_samples():
    return [
        {
            "index": i,
            "npy_name": row["preprocessed_name"],
            "true_label": row[SPECIES_COLUMN],
            "audio_path": row["local_path"]
        }
        for i, row in test_df.iterrows()
    ]


@app.get("/audio/{index}")
def get_audio(index: int):
    return FileResponse(test_df.loc[index, "local_path"])


@app.get("/evaluate")
def evaluate(index: int):
    row = test_df.loc[index:index]  # keep as DataFrame
    sample_ds = AudioFeatureDataset(row, species_to_idx)  # single-sample dataset
    x, _ = sample_ds[0]  # get the first (and only) sample

    x = x.unsqueeze(0).to(classifier.device)
    classifier.model.to(classifier.device)

    with torch.no_grad():
        outputs = classifier.model(x)
        logits = outputs["clipwise_output"]
        pred_idx = torch.argmax(logits, dim=1).item()

    predicted_label = species_list[pred_idx]

    return {
        "true_label": row.iloc[0][SPECIES_COLUMN],
        "predicted_label": predicted_label
    }

@app.get("/features/{index}")
def list_arrays(index: int):
    row = test_df.loc[index]
    data = np.load(row["features_path"])
    return {"features": list(data.keys())}

@app.get("/features_plot/{index}")
def get_array_plot(index: int, feature_name: str):
    row = test_df.loc[index]
    data = np.load(row["features_path"])
    
    if feature_name not in data:
        return {"error": "features not found"}

    x = data[feature_name]

    fig, ax = plt.subplots(figsize=(8, 4))

    if x.ndim == 1:
        ax.plot(x)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
    elif x.ndim == 2:
        ax.imshow(x.T, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature")
    elif x.ndim == 3:
        # Take the first slice along the first dimension
        ax.imshow(x[0].T, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel("Time")
        ax.set_ylabel("Feature")
        ax.set_title("Showing first slice of 3D array")
    else:
        ax.text(0.5, 0.5, f"Cannot display array with ndim={x.ndim}", 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
