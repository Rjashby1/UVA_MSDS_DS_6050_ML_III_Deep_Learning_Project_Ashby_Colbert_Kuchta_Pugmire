"""
data_prep.py
============
One-time preprocessing pipeline for the ISIC 2019 / MILK10K project.

Run ONCE after download_data.py. Outputs go to DATA/processed/.
Training scripts then load from processed/ -- never reprocess 25k images again.

Usage:
    python scripts/data_prep.py

Outputs:
    DATA/processed/isic2019/train.csv           <- ~85% of ISIC 2019 train folder, stratified
    DATA/processed/isic2019/val.csv             <- ~15% of ISIC 2019 train folder, stratified
    DATA/processed/isic2019/test.csv            <- official ISIC 2019 test set (independently curated)
    DATA/processed/milk10k/external_val.csv     <- MILK10K external validation cohort (strictly siloed)
    DATA/processed/metadata_scaler.pkl          <- fitted StandardScaler (age) — fit on train only
    DATA/processed/metadata_ohe.pkl             <- fitted OneHotEncoder (sex, site) — fit on train only
    DATA/processed/class_weights.json           <- inverse-frequency weights per class label

Notes:
    - The official ISIC 2019 test set is used as our internal test split (Option A).
      Competition-specific columns (UNK, score_weight, validation_weight) are dropped.
    - MILK10K test folder is NOT used — it has no ground truth labels.
    - MILK10K classes BEN_OTH and MAL_OTH are excluded — no equivalent in ISIC 2019 training data.
    - Image paths are stored as relative paths from project root for cross-platform compatibility
      (works on Windows, Mac, and Rivanna/Linux).
    - Patient-level leakage cannot be fully prevented as ISIC 2019 does not provide patient IDs.
      Lesion ID is retained for traceability. This is a known limitation documented in the literature.
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "DATA"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ISIC 2019 — training folder (labeled, used for train/val)
ISIC_TRAIN_IMG  = DATA / "isic2019" / "train" / "images" / "ISIC_2019_Training_Input"
ISIC_TRAIN_META = DATA / "isic2019" / "train" / "ISIC_2019_Training_Metadata.csv"
ISIC_TRAIN_GT   = DATA / "isic2019" / "train" / "ISIC_2019_Training_GroundTruth.csv"

# ISIC 2019 — official test set (used as our internal test split)
ISIC_TEST_IMG   = DATA / "isic2019" / "test" / "images" / "ISIC_2019_Test_Input"
ISIC_TEST_META  = DATA / "isic2019" / "test" / "ISIC_2019_Test_Metadata.csv"
ISIC_TEST_GT    = DATA / "isic2019" / "test" / "ISIC_2019_Test_GroundTruth.csv"

# MILK10K — training folder only (test folder has no ground truth)
MILK_TRAIN_IMG  = DATA / "milk10k" / "train" / "images" / "MILK10k_Training_Input"
MILK_TRAIN_META = DATA / "milk10k" / "train" / "MILK10k_Training_Metadata.csv"
MILK_TRAIN_GT   = DATA / "milk10k" / "train" / "MILK10k_Training_GroundTruth.csv"

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRAC    = 0.15   # 15% of ISIC 2019 training folder → internal validation

# Metadata columns used for the MLP branch
# Note: ISIC 2019 calls the body site column 'anatom_site_general'
#       MILK10K calls it 'site' — both are renamed to 'anatom_site_general' during preprocessing
META_CATEGORICAL = ["sex", "anatom_site_general"]
META_NUMERICAL   = ["age_approx"]

# Malignant class labels per dataset
ISIC_MALIGNANT = {"MEL", "BCC", "SCC", "AK"}
MILK_MALIGNANT = {"MEL", "BCC", "SCC", "AK", "SCCKA"}

# MILK10K classes excluded — no equivalent in ISIC 2019 training data
MILK_EXCLUDE = {"BEN_OTH", "MAL_OTH"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build ISIC 2019 master DataFrame from training folder
#   Merge ground-truth labels + metadata, resolve image paths
# ══════════════════════════════════════════════════════════════════════════════
def build_isic2019_df() -> pd.DataFrame:
    """
    Loads the ISIC 2019 training folder.
    Returns a DataFrame with columns:
        image_id, image_path (relative), label, is_malignant,
        age_approx, sex, anatom_site_general, lesion_id
    """
    print("\n[1/5] Building ISIC 2019 master dataframe from training folder...")

    gt   = pd.read_csv(ISIC_TRAIN_GT)
    meta = pd.read_csv(ISIC_TRAIN_META)

    # Ground-truth is one-hot encoded across diagnosis columns.
    # Convert to a single 'label' string (e.g. "MEL", "NV", "BCC" ...)
    diag_cols = [c for c in gt.columns if c != "image"]
    gt["label"] = gt[diag_cols].idxmax(axis=1)

    # Merge on image ID
    df = gt[["image", "label"]].merge(meta, on="image", how="left")
    df = df.rename(columns={"image": "image_id"})

    # Add malignant flag
    df["is_malignant"] = df["label"].isin(ISIC_MALIGNANT).astype(int)

    # Resolve relative image paths
    def find_image(img_id: str) -> str:
        p = ISIC_TRAIN_IMG / f"{img_id}.jpg"
        if p.exists():
            return str(p.relative_to(ROOT))
        matches = list(ISIC_TRAIN_IMG.rglob(f"{img_id}.jpg"))
        return str(matches[0].relative_to(ROOT)) if matches else ""

    df["image_path"] = df["image_id"].apply(find_image)
    missing = (df["image_path"] == "").sum()
    if missing:
        print(f"  WARNING: {missing} images not found on disk — they will be dropped.")
    df = df[df["image_path"] != ""].reset_index(drop=True)

    print(f"  Total images after path resolution: {len(df)}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Stratified Train / Val split
#   85% train, 15% val — both from the ISIC 2019 training folder.
#   Internal test set comes from the official ISIC 2019 test set (Step 2b).
# ══════════════════════════════════════════════════════════════════════════════
def split_isic2019(df: pd.DataFrame):
    """
    Returns (train_df, val_df)

    Split ratios (of the ISIC 2019 training folder):
        train : val  ≈  85 : 15

    Internal test set is the official ISIC 2019 test set,
    loaded separately in build_isic2019_test_df().
    """
    print("\n[2/5] Stratified train/val split on ISIC 2019 training folder...")

    train, val = train_test_split(
        df, test_size=VAL_FRAC, stratify=df["label"], random_state=RANDOM_SEED
    )

    for name, subset in [("train", train), ("val", val)]:
        print(f"  {name:14s}: {len(subset):6d} images")

    return train.reset_index(drop=True), val.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b — Load official ISIC 2019 test set as our internal test split
#   This is independently curated by ISIC — a cleaner separation than
#   carving off a portion of the training folder ourselves.
#   Competition-specific columns (UNK, score_weight, validation_weight) are dropped.
# ══════════════════════════════════════════════════════════════════════════════
def build_isic2019_test_df() -> pd.DataFrame:
    """
    Loads the official ISIC 2019 test set.
    Returns a DataFrame with the same columns as the training splits.
    """
    print("\n[2b/5] Loading official ISIC 2019 test set...")

    gt   = pd.read_csv(ISIC_TEST_GT)
    meta = pd.read_csv(ISIC_TEST_META)

    # Drop competition-specific columns not present in training data
    drop_cols = ["UNK", "score_weight", "validation_weight"]
    gt = gt.drop(columns=[c for c in drop_cols if c in gt.columns])

    # Convert one-hot to single label string
    diag_cols = [c for c in gt.columns if c != "image"]
    gt["label"] = gt[diag_cols].idxmax(axis=1)

    df = gt[["image", "label"]].merge(meta, on="image", how="left")
    df = df.rename(columns={"image": "image_id"})

    # Add malignant flag
    df["is_malignant"] = df["label"].isin(ISIC_MALIGNANT).astype(int)

    # Resolve relative image paths
    def find_image(img_id: str) -> str:
        p = ISIC_TEST_IMG / f"{img_id}.jpg"
        if p.exists():
            return str(p.relative_to(ROOT))
        matches = list(ISIC_TEST_IMG.rglob(f"{img_id}.jpg"))
        return str(matches[0].relative_to(ROOT)) if matches else ""

    df["image_path"] = df["image_id"].apply(find_image)
    missing = (df["image_path"] == "").sum()
    if missing:
        print(f"  WARNING: {missing} test images not found on disk — dropping.")
    df = df[df["image_path"] != ""].reset_index(drop=True)

    print(f"  Official test set loaded: {len(df)} images")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Fit metadata encoders ON TRAINING SET ONLY, then transform all splits
#
#   WHY this order matters:
#     - StandardScaler / OneHotEncoder must be fit only on TRAIN data.
#     - Val, test, and MILK10K data are transformed using TRAIN-fitted encoders.
#     - This prevents data leakage (test statistics bleeding into training).
# ══════════════════════════════════════════════════════════════════════════════
def fit_and_encode_metadata(train_df, val_df, test_df, milk_df=None):
    """
    Fits encoders on train_df, transforms all splits.
    Saves encoders to disk for later use by training scripts.
    Adds encoded columns to each dataframe and returns them.
    """
    print("\n[3/5] Fitting metadata encoders on training set only...")

    all_dfs = {"train": train_df, "val": val_df, "test": test_df}
    if milk_df is not None:
        all_dfs["milk"] = milk_df

    # ── Numerical: StandardScaler for age ─────────────────────────────────
    scaler = StandardScaler()
    train_median = train_df[META_NUMERICAL].median()

    train_df["age_scaled"] = scaler.fit_transform(
        train_df[META_NUMERICAL].fillna(train_median)
    )
    for name, df in all_dfs.items():
        if name == "train":
            continue
        df["age_scaled"] = scaler.transform(
            df[META_NUMERICAL].fillna(train_median)
        )

    # ── Categorical: OneHotEncoder for sex + anatomical site ──────────────
    # Fill NaN with "unknown" so OHE doesn't break on missing values
    for df in all_dfs.values():
        for col in META_CATEGORICAL:
            if col not in df.columns:
                df[col] = "unknown"
            else:
                df[col] = df[col].fillna("unknown").str.lower().str.strip()

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    train_ohe = ohe.fit_transform(train_df[META_CATEGORICAL])
    ohe_cols  = ohe.get_feature_names_out(META_CATEGORICAL)

    for name, df in all_dfs.items():
        encoded = ohe.transform(df[META_CATEGORICAL]) if name != "train" else train_ohe
        for i, col in enumerate(ohe_cols):
            df[col] = encoded[:, i]

    print(f"  OHE columns created: {list(ohe_cols)}")
    print(f"  Numerical columns:   {META_NUMERICAL} → age_scaled")

    # ── Save encoders ─────────────────────────────────────────────────────
    scaler_path = PROCESSED / "metadata_scaler.pkl"
    ohe_path    = PROCESSED / "metadata_ohe.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(ohe_path, "wb") as f:
        pickle.dump(ohe, f)
    print(f"  Saved scaler → {scaler_path}")
    print(f"  Saved OHE    → {ohe_path}")

    return all_dfs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Compute class weights for weighted cross-entropy loss
#
#   Inverse-frequency weighting:
#     weight[c] = N_total / (N_classes * count[c])
#
#   Rare malignant classes get a much higher loss penalty than the majority
#   class (NV), preventing the model from collapsing into predicting only
#   the most common class. Weights are plugged into CrossEntropyLoss(weight=...)
#   when the training script is written.
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(train_df: pd.DataFrame) -> dict:
    print("\n[4/5] Computing inverse-frequency class weights...")

    counts    = train_df["label"].value_counts()
    n_total   = len(train_df)
    n_classes = len(counts)
    weights   = {}

    for cls, cnt in counts.items():
        w = n_total / (n_classes * cnt)
        weights[cls] = round(w, 4)
        print(f"  {cls:35s}: count={cnt:5d}  weight={w:.4f}")

    out_path = PROCESSED / "class_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"  Saved class weights → {out_path}")
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build MILK10K external validation cohort
#
#   MILK10K image structure:
#     milk10k/train/images/MILK10k_Training_Input/
#       IL_0006205/
#           ISIC_XXXXXXX.jpg    ← dermoscopic image
#           ISIC_YYYYYYY.jpg    ← clinical close-up
#
#   We filter metadata to dermoscopic rows only before merging,
#   giving us one clean row per lesion.
#   Both image paths are stored for future multi-modal experiments.
#
#   NOTE: MILK10K test folder is NOT used — it has no ground truth labels.
#         BEN_OTH and MAL_OTH classes are excluded — no equivalent in ISIC 2019.
#         MONET columns are retained in the CSV but not used by our MLP branch,
#         as they have no equivalent in ISIC 2019.
# ══════════════════════════════════════════════════════════════════════════════
def build_milk10k_df() -> pd.DataFrame:
    """
    Loads the MILK10K training folder as our external validation cohort.
    Returns a DataFrame with columns:
        lesion_id, label, is_malignant, image_path_derm, image_path_clinic,
        age_approx, sex, anatom_site_general, age_scaled, [OHE columns]
    """
    print("\n[5/5] Building MILK10K external validation dataframe...")

    gt   = pd.read_csv(MILK_TRAIN_GT)
    meta = pd.read_csv(MILK_TRAIN_META)

    # Convert one-hot to single label string
    if "label" not in gt.columns:
        diag_cols = [c for c in gt.columns if c != "lesion_id"]
        gt[diag_cols] = gt[diag_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        gt["label"] = gt[diag_cols].idxmax(axis=1)

    # Drop classes with no equivalent in ISIC 2019 training data
    excluded = gt[gt["label"].isin(MILK_EXCLUDE)]
    if len(excluded) > 0:
        print(f"  Dropping {len(excluded)} samples with excluded classes: {MILK_EXCLUDE}")
    gt = gt[~gt["label"].isin(MILK_EXCLUDE)]

    # Filter to dermoscopic rows only — one row per lesion
    meta = meta[meta["image_type"] == "dermoscopic"].drop_duplicates(subset="lesion_id")

    df = gt[["lesion_id", "label"]].merge(meta, on="lesion_id", how="left")

    # Rename site column to match ISIC 2019 naming convention
    df = df.rename(columns={"site": "anatom_site_general"})

    # Add malignant flag
    df["is_malignant"] = df["label"].isin(MILK_MALIGNANT).astype(int)

    # Resolve both image paths per lesion folder
    def resolve_pair(lesion_id: str):
        folder = MILK_TRAIN_IMG / lesion_id
        if not folder.exists():
            matches = list(MILK_TRAIN_IMG.rglob(lesion_id))
            folder = matches[0] if matches else None
        if folder is None or not folder.is_dir():
            return "", ""
        imgs = sorted(folder.glob("*.jpg"))
        path_a = str(imgs[0].relative_to(ROOT)) if len(imgs) > 0 else ""
        path_b = str(imgs[1].relative_to(ROOT)) if len(imgs) > 1 else ""
        return path_a, path_b

    df[["image_path_derm", "image_path_clinic"]] = df["lesion_id"].apply(
        lambda sid: pd.Series(resolve_pair(sid))
    )

    missing = (df["image_path_derm"] == "").sum()
    if missing:
        print(f"  WARNING: {missing} MILK10K samples with no images found — dropping.")
    df = df[df["image_path_derm"] != ""].reset_index(drop=True)

    print(f"  MILK10K samples loaded: {len(df)}")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  data_prep.py — One-time preprocessing pipeline")
    print("=" * 60)

    # Step 1: Load + merge ISIC 2019 training folder
    isic_df = build_isic2019_df()

    # Step 2: Split training folder into train/val (85/15)
    train_df, val_df = split_isic2019(isic_df)

    # Step 2b: Load official ISIC 2019 test set as internal test split
    test_df = build_isic2019_test_df()

    # Step 5: Load MILK10K external validation cohort
    milk_df = build_milk10k_df()

    # Step 3: Fit + encode metadata (encoders fit on train only, applied to all)
    all_dfs = fit_and_encode_metadata(train_df, val_df, test_df, milk_df)

    # Step 4: Compute class weights from training set
    compute_class_weights(all_dfs["train"])

    # ── Save processed CSVs ───────────────────────────────────────────────
    print("\n[Saving] Writing processed CSVs to DATA/processed/...")
    isic_out = PROCESSED / "isic2019"
    milk_out = PROCESSED / "milk10k"
    isic_out.mkdir(parents=True, exist_ok=True)
    milk_out.mkdir(parents=True, exist_ok=True)

    all_dfs["train"].to_csv(isic_out / "train.csv",       index=False)
    all_dfs["val"].to_csv(isic_out / "val.csv",           index=False)
    all_dfs["test"].to_csv(isic_out / "test.csv",         index=False)
    all_dfs["milk"].to_csv(milk_out / "external_val.csv", index=False)

    print("\n  Done! Processed files written:")
    for p in sorted(PROCESSED.rglob("*.csv")):
        print(f"    {p.relative_to(ROOT)}")
    for p in sorted(PROCESSED.rglob("*.pkl")):
        print(f"    {p.relative_to(ROOT)}")
    for p in sorted(PROCESSED.rglob("*.json")):
        print(f"    {p.relative_to(ROOT)}")

    print("\n  All downstream training scripts should load from DATA/processed/")
    print("  Never re-run this script unless you re-download the raw data.")
    print("=" * 60)


if __name__ == "__main__":
    main()