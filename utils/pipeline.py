import os
import shutil
from pathlib import Path
import pandas as pd


PIPELINE_ORDER = [
    "get_data",
    "preprocessing",
    "features",
    "split_data",
    "train"
]


# --------------------------------------------------------------
# Helper: delete files/directories of specific pipeline steps
# --------------------------------------------------------------
def cleanup_after(step):
    """Remove outputs produced AFTER a given step."""
    index = PIPELINE_ORDER.index(step)

    steps_to_clean = PIPELINE_ORDER[index + 1:]

    for s in steps_to_clean:
        if s == "preprocessing":
            shutil.rmtree("preprocessed", ignore_errors=True)

        elif s == "features":
            shutil.rmtree("features", ignore_errors=True)

        elif s == "split_data":
            if Path("dataset_splits.csv").exists():
                Path("dataset_splits.csv").unlink()

        elif s == "train":
            if Path("cnn14_finetuned.pth").exists():
                Path("cnn14_finetuned.pth").unlink()

    print(f"Cleanup completed for: {steps_to_clean}")


# --------------------------------------------------------------
# Step checkers
# --------------------------------------------------------------
def is_step_done(step):
    """Returns True if outputs of the step exist."""
    if step == "get_data":
        return Path("dataset").exists() and list(Path("dataset").rglob("*.wav"))

    if step == "preprocessing":
        return Path("preprocessed").exists() and list(Path("preprocessed").rglob("*.npy"))

    if step == "features":
        return Path("features").exists() and list(Path("features").rglob("*.npz"))

    if step == "split_data":
        return Path("dataset_splits.csv").exists()

    if step == "train":
        return Path("cnn14_finetuned.pth").exists()

    return False


# --------------------------------------------------------------
# Main pipeline controller
# --------------------------------------------------------------
def check_and_run_pipeline():
    """Determine which steps must run, respecting strict order."""
    
    steps_needed = []
    found_missing = False

    for step in PIPELINE_ORDER:

        if not is_step_done(step):
            # first missing step found — clean after and mark as needed
            if not found_missing:
                cleanup_after(step)
                found_missing = True

            steps_needed.append(step)

    return steps_needed


# --------------------------------------------------------------
# Execution of each step
# --------------------------------------------------------------
def run_pipeline_step(step_name):
    if step_name == "get_data":
        from utils.get_data import main as get_data_main
        print("=== Downloading dataset ===")
        get_data_main()

    elif step_name == "preprocessing":
        from utils.preprocessing import preprocess_dataset
        print("=== Preprocessing audio files ===")
        preprocess_dataset()

    elif step_name == "features":
        from utils.features import process_preprocessed_dir
        print("=== Extracting features ===")
        process_preprocessed_dir()

    elif step_name == "split_data":
        print("=== Creating train/val/test splits ===")
        os.system("python utils/split_data.py")

    elif step_name == "train":
        from utils.train import main
        print("=== Training model ===")
        main()


# --------------------------------------------------------------
# Full pipeline run
# --------------------------------------------------------------
def run_full_pipeline():
    steps = check_and_run_pipeline()

    if not steps:
        print("All pipeline steps completed. Ready to test!")
        return True

    print(f"Steps needed: {', '.join(steps)}")

    for step in steps:
        run_pipeline_step(step)
        print(f"✓ {step} completed\n")

    print("Pipeline complete!")
    return True
