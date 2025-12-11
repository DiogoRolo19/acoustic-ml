import pandas as pd
from pathlib import Path

# --------------------------
# CONFIG
# --------------------------
dataset_root = Path("dataset")
preprocessed_root = Path("preprocessed")
features_root = Path("features")
output_csv = Path("dataset_splits.csv")

# --------------------------
# 1) Load all metadata.csv files under dataset/**
# --------------------------
meta_parts = []
for meta_file in dataset_root.rglob("metadata.csv"):
    df_part = pd.read_csv(meta_file)
    # normalize metadata 'file' by removing .wav only
    df_part['file_no_ext'] = df_part['file'].astype(str).str.replace(r'(?i)\.wav$', '', regex=True)
    df_part['meta_source_dir'] = str(meta_file.parent)
    meta_parts.append(df_part)

if not meta_parts:
    raise RuntimeError("No metadata.csv files found under dataset/")

meta_df = pd.concat(meta_parts, ignore_index=True)

# --------------------------
# 2) Build metadata lookup: keep ONLY the first row for each base name
# --------------------------
meta_index = {}
for _, row in meta_df.iterrows():
    key = str(row['file_no_ext'])
    if key not in meta_index:     # keep FIRST occurrence
        meta_index[key] = row

# --------------------------
# 3) Scan features/** for all .npy files
# --------------------------
npy_rows = []

for npy_path in preprocessed_root.rglob("*.npy"):
    npy_name = npy_path.stem  # "6102500A_chunk0"
    prefix = npy_name.split("_", 1)[0]  # "6102500A"

    relative_path = npy_path.relative_to(preprocessed_root)
    npz_path = features_root / relative_path.with_suffix(".npz")

    meta = meta_index.get(prefix, None)

    row = {"preprocessed_path": str(npy_path), "features_path": str(npz_path), "preprocessed_name": npy_name, "meta_key": prefix}

    if meta is None:
        # no metadata match — fill metadata columns as NA
        for col in meta_df.columns:
            if col not in row:
                row[col] = pd.NA
    else:
        # copy FIRST matching metadata row
        for col in meta_df.columns:
            row[col] = meta[col]

    npy_rows.append(row)

# one-line-per-npy
df = pd.DataFrame(npy_rows)

# --------------------------
# 4) Fix date + group values
# --------------------------
df['DA_parsed'] = pd.to_datetime(df['DA'], errors='coerce')

df['GA_first'] = df['GA']
df['RG_first'] = df['RG']

df['group'] = df['GA_first'].astype(str) + "_" + df['RG_first'].astype(str)

# --------------------------
# 5) Chronological group ordering
# --------------------------
group_time = (
    df.groupby('group', dropna=False)['DA_parsed']
      .apply(lambda s: s.dropna().mean() if len(s.dropna()) > 0 else pd.NaT)
      .reset_index(name='DA_mean')
      .sort_values('DA_mean')
      .reset_index(drop=True)
)

n_groups = len(group_time)
if n_groups == 0:
    raise RuntimeError("No groups found.")

ideal_train_end = 0.7 * n_groups
ideal_val_end   = 0.85 * n_groups

def boundary(idx_float, min_allowed):
    idx = int(idx_float)
    return max(idx, min_allowed)

train_end = boundary(ideal_train_end, 1)
val_end   = boundary(ideal_val_end, train_end + 1)

if val_end >= n_groups:
    if train_end < n_groups - 1:
        val_end = n_groups - 1
    else:
        train_end = max(1, n_groups - 2)
        val_end   = n_groups - 1

train_groups = group_time.iloc[:train_end]['group'].tolist()
val_groups   = group_time.iloc[train_end:val_end]['group'].tolist()
test_groups  = group_time.iloc[val_end:]['group'].tolist()

# --------------------------
# 6) Assign split
# --------------------------
def assign_split(g):
    if g in train_groups:
        return 'train'
    if g in val_groups:
        return 'val'
    return 'test'

df['split'] = df['group'].apply(assign_split)

# --------------------------
# 7) Save result
# --------------------------
df.to_csv(output_csv, index=False)
print(f"Saved {output_csv} with {len(df)} rows.")
