# Technical Assessment — My Approach and Key Decisions

## 1. Dataset Selection & Initial Preparation

For this challenge, I selected the **Watkins Marine Mammal Sound Database**, focusing on the **best_of subset**, which contains approximately **1,700 high-quality marine mammal recordings**. This dataset provides a diverse set of species and includes useful metadata such as location, recording equipment, and timestamps.

After downloading both the audio and metadata, I performed an initial scan of the dataset:

**Dataset statistics:**

- Number of files: 1,697  
- Sampling rate: min = 600 Hz, max = 166,600 Hz, mean ≈ 57,788 Hz  
- Duration: min = 0.05 s, max = 1,260 s, mean ≈ 10.43 s  

No major acquisition issues were found, but the dataset presents significant variability in sampling rates and durations.

---

## 2a. Preprocessing Decisions

### Audio Duration Standardization

To feed a consistent input to the model, I standardized all samples to **10-second audio segments**:

- Very short files were discarded.  
- Very long files were split into 10-second chunks.  
- Final partial chunks shorter than 10 seconds were padded with zeros at the end.  

### Resampling and Normalization

Since the average sampling rate was close to 58 kHz, I chose **44.1 kHz** as the resampling target — a common, well-supported standard that reduces computational load.  

I also:

- Normalized amplitude per sample to reduce differences caused by recording gain.  
- Retained other audio processing defaults, given the limited time and the challenge scope.  

This produced a clean, uniform set of training-ready audio segments.

---

## 2b. Feature Extraction

After preprocessing, I extracted features to represent the audio segments in ways suitable for machine learning:

### Waveform (1D features)

- Raw 10-second audio samples normalized per file.  
- Captures the time-domain signal for transient detection.

### Spectrogram / Mel-spectrogram (2D features)

- Transformed audio segments using STFT → Mel-spectrogram.  
- Default parameters were used (`n_fft`, `hop_length`, `n_mels`) due to time constraints and limited acoustic expertise.  
- Sample rate for all features: 44.1 kHz, close to the dataset mean (~57 kHz) to reduce resampling artifacts.

### Aggregate Features (1D statistics)

- Root-mean-square (RMS) energy  
- Spectral centroid  
- Bandwidth  
- Zero-crossing rate (ZCR)

**Justification for choices:**

- Combination of 1D waveform and 2D spectrogram features captures both temporal patterns and spectral content.  
- Default STFT/Mel parameters provide a reasonable baseline; hyperparameter tuning could improve performance if more time/resources were available.  
- Simple features ensure compatibility with pretrained architectures (like CNN14) and reduce computational load for a challenge demo.

---

## 3. Data Splitting Strategy

Main goals were to avoid:

- Data leakage  
- Temporal bias  
- Overfitting to specific recording equipment or locations  
- Class imbalance issues  

### Grouping Strategy

Recordings were grouped by:

- Location  
- Recording equipment  

Within each group, the mean recording date was computed and groups were sorted chronologically.

### Split Approach

Groups (not individual samples) were split into:

- **Train:** ~75%  
- **Validation:** ~10%  
- **Test:** ~15%  

This ensures:

- No leakage between recordings from the same place/equipment  
- Temporal consistency (train on older data, validate/test on newer data)  
- Reduced risk of overfitting to specific hydrophones or environmental conditions  

Class imbalance was not handled at this stage — it was addressed later in the loss function.

---

## 4. Model Architecture Selection

Initially, the plan was to train a full model from scratch using:

- 1D waveform features  
- 2D spectrogram features  

Due to limited computational resources, I shifted to a more practical solution.

### Pretrained Model Approach

- Selected **CNN14**, a well-known pretrained audio classification model.  
- Steps:  
  - Load pretrained CNN14 base  
  - Replace final classification layer to match my dataset  
  - Fine-tune final two layers instead of the entire model  
  - Use a **class-weighted loss** to handle class imbalance  

### Class Simplification

- Original dataset contains many fine-grained species and subspecies  
- Collapsed taxonomy into **5 main species groups** (e.g., dolphins vs. whales) to improve learning and reduce overfitting risk  
- Full classification across subspecies would require more sophisticated loss functions and more compute

### Training Handling

- Saved model checkpoints every 10 epochs (no early stopping implemented by design)  
- Allows manual rollback in case overfitting appears  
- Early stopping was not essential due to small-scale training

---

## 5. Web-Based Visualization

- Built a minimal but functional web interface (no CSS styling, focusing on content)  
- Interface allows users to explore:  
  - Waveforms  
  - Spectrograms  
  - Other useful acoustic features  

**Important Note:**

- Individual 10-second chunks were not saved as separate `.wav` files  
- The interface always loads the original full recording:  
  - Chunk 1 = seconds 0–10  
  - Chunk 2 = seconds 10–20  
  - etc.  
- If audio playback per chunk were required, each segment would need to be generated and stored separately

---

## 6. Summary of Key Decisions

- Chose a marine bioacoustics dataset with rich metadata  
- Standardized audio to **10-second, 44.1 kHz, normalized segments**  
- Split data by **location + equipment**, sorted by time to avoid leakage and bias  
- Used **CNN14 pretrained**, fine-tuned final layers with class-weighted loss  
- Simplified taxonomy to **5 species** for practical training  
- Implemented a lightweight web visualization to explore signals and metadata  
- Documented all choices without over-engineering beyond challenge scope
