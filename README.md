# Acoustic ML Pipeline — Watkins Marine Mammal Sounds

## Goal
This project demonstrates a complete acoustic ML pipeline:
- Download and preprocess marine mammal audio recordings.
- Extract features (waveform, spectrogram, aggregate statistics).
- Train/fine-tune a model for species classification.
- Explore the dataset and predictions through a lightweight web interface.

---

## Getting Started

### 1. Clone the repository
```bash
git clone <https://github.com/DiogoRolo19/acoustic-ml/tree/master>
cd <acoustic-ml>
```

### 2. Install requirements
```bash
python3.13 -m pip install -r requirements.txt
```

### 3. Download model and weights
- https://drive.google.com/drive/folders/1D3vsUDSofWpdNcOs-NTfzK9jgQBf_tkT?usp=drive_link


- Run the main pipeline:
```bash
python main.py
```

### 4. Launch web visualization
```bash
python api.py
```

- Open `index.html` in your browser to explore the dataset, view waveforms, spectrograms, and metadata.

---

## Notes
- Audio segments are standardized to 10-second chunks.
- The web interface shows chunks relative to the original recording (chunk 1 = seconds 0–10, chunk 2 = 10–20, etc.).

