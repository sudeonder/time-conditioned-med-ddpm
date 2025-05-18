#!/usr/bin/env bash
# organize_data.sh
# Usage: bash organize_data.sh /path/to/MU-Glioma-Post /path/to/med-ddpm

# 1) Source and target roots
DATA_ROOT="$1"                   # e.g. /content/data/Dataset_MU-Glioma-Post/MU-Glioma-Post
REPO_ROOT="$2"                   # e.g. /content/med-ddpm
BRATS_ROOT="$REPO_ROOT/dataset/brats2021"

if [[ -z "$DATA_ROOT" || -z "$REPO_ROOT" ]]; then
  echo "Usage: bash organize_data.sh <MU-Glioma-Post-root> <med-ddpm-root>"
  exit 1
fi

echo "Data root:   $DATA_ROOT"
echo "Repo root:   $REPO_ROOT"
echo "BraTS root:  $BRATS_ROOT"

# 2) Make BraTS folders
mkdir -p "$BRATS_ROOT"/{seg,t1,t1ce,t2,flair}

# 3) Symlink tumor masks into seg/PatientID_xx/Timepoint_n/
echo "Linking masks..."
find "$DATA_ROOT" -type f -name '*_tumorMask.nii.gz' | while read -r MASK; do
  PAT=$(basename "$(dirname "$(dirname "$MASK")")")
  TP=$(basename "$(dirname "$MASK")")
  DEST="$BRATS_ROOT/seg/$PAT/$TP"
  mkdir -p "$DEST"
  ln -sf "$MASK" "$DEST/"
done

# 4) Symlink T1c scans into t1/PatientID_xx/Timepoint_n/
echo "Linking T1c scans..."
find "$DATA_ROOT" -type f -name '*brain_t1c.nii.gz' | while read -r IMG; do
  PAT=$(basename "$(dirname "$(dirname "$IMG")")")
  TP=$(basename "$(dirname "$IMG")")
  DEST="$BRATS_ROOT/t1/$PAT/$TP"
  mkdir -p "$DEST"
  ln -sf "$IMG" "$DEST/"
done

# 5) Optional: mirror t1/ into t1ce/, t2/, flair/
echo "Mirroring t1/ into t1ce/, t2/, flair/..."
for MOD in t1ce t2 flair; do
  mkdir -p "$BRATS_ROOT/$MOD"
  find "$BRATS_ROOT/t1" -type f | while read -r F; do
    SUB="${F#*/seg/}"       # cut off everything before seg/
    TARGET_DIR="$BRATS_ROOT/$MOD/$(dirname "${SUB#*/}")"
    mkdir -p "$TARGET_DIR"
    ln -sf "$F" "$TARGET_DIR/"
  done
done

echo "Done. BraTS‐style dataset is under $BRATS_ROOT"
