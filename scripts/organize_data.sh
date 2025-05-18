#!/usr/bin/env bash
# flatten_data.sh
# Usage: bash flatten_data.sh <MU-Glioma-Post-root> <med-ddpm-root>
#
# Example:
#   bash flatten_data.sh /content/data/Dataset_MU-Glioma-Post/MU-Glioma-Post /content/med-ddpm

DATA_ROOT="$1"      # e.g. /content/data/.../MU-Glioma-Post
REPO_ROOT="$2"      # e.g. /content/med-ddpm
BRATS_DIR="$REPO_ROOT/dataset/brats2021"

if [[ -z "$DATA_ROOT" || -z "$REPO_ROOT" ]]; then
  echo "Usage: bash flatten_data.sh <MU-Glioma-Post-root> <med-ddpm-root>"
  exit 1
fi

echo "Flattening data from $DATA_ROOT into $BRATS_DIR"

# 1) Prepare target folders
mkdir -p "$BRATS_DIR/seg"
mkdir -p "$BRATS_DIR/t1"

# 2) Link all tumorMask files into seg/
echo "Linking all *_tumorMask.nii.gz into $BRATS_DIR/seg/"
find "$DATA_ROOT" -type f -name '*_tumorMask.nii.gz' | while read -r MASK; do
  BAS=$(basename "$MASK")
  ln -sf "$MASK" "$BRATS_DIR/seg/$BAS"
done

# 3) Link all brain_t1c files into t1/
echo "Linking all *brain_t1c.nii.gz into $BRATS_DIR/t1/"
find "$DATA_ROOT" -type f -name '*brain_t1c.nii.gz' | while read -r IMG; do
  BAS=$(basename "$IMG")
  ln -sf "$IMG" "$BRATS_DIR/t1/$BAS"
done

echo "Done! You now have:"
echo "  $BRATS_DIR/seg/ -> $(ls -1 "$BRATS_DIR/seg" | wc -l) masks"
echo "  $BRATS_DIR/t1/  -> $(ls -1 "$BRATS_DIR/t1" | wc -l) images"
