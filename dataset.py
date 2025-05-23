# dataset.py
# -*- coding:utf-8 -*-
import os
import re
import glob

import numpy as np
import pandas as pd
import torch
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
def fourier_time_emb(dt_norm: float, L: int = 6):
    """
    Positional encoding of a single scalar dt_norm in [0,1]:
    returns a vector of length 2*L: [sin(2^0 π dt), cos(2^0 π dt), ..., sin(2^{L-1} π dt), cos(...)]
    """
    freqs = 2.0 ** np.arange(L)                        # [1,2,4,...,2^{L-1}]
    angles = np.outer(freqs, 2 * np.pi * dt_norm)     # shape (L,)
    sincos = np.concatenate([np.sin(angles), np.cos(angles)], axis=0)
    return sincos.astype(np.float32)                   # shape (2L,)

# -----------------------------------------------------------------------------
# single resize operator for both cond and target
RESIZE = tio.Resize((192, 192, 144))

def cond_img_resize(cond: np.ndarray, img: np.ndarray):
    """
    Resize + reorder volumes so that:
      cond → (C_cond, D=144, H=192, W=192)
      img  → (1,      144,    192,    192)
    where C_cond = 1 + 2L (mask channel + fourier channels).
    """
    # cond/img are (C, X=240, Y=240, Z=155)
    cond_r = RESIZE(tio.ScalarImage(tensor=cond)).data.numpy()
    img_r  = RESIZE(tio.ScalarImage(tensor=img )).data.numpy()
    # swap to (C, D, H, W)
    cond_out = cond_r.transpose(0, 3, 1, 2)
    img_out  = img_r.transpose(0, 3, 1, 2)
    return cond_out, img_out

# -----------------------------------------------------------------------------
class MUTimeConditionedDataset(Dataset):
    """
    Flat-folder loader for MU-Glioma-Post with time-conditioning.
    Returns dicts with
      'input':  FloatTensor[C_cond, D, H, W]
      'target': FloatTensor[1,      D, H, W]
    where C_cond = 1 + 2L (one binary mask channel + fourier time embedding).
    """
    def __init__(self,
                 mask_dir: str,
                 img_dir: str,
                 clinical_xlsx: str,
                 max_weeks: float,
                 transform=cond_img_resize
                 ):
        # --- 1) load clinical excel ---
        xls        = pd.ExcelFile(clinical_xlsx)
        sheet      = xls.sheet_names[1]
        df         = pd.read_excel(xls, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        df.set_index('Patient_ID', inplace=True)
        self.clin_df = df

        # --- 2) map Timepoint_n → column name ---
        self.tp_col_map = {
            int(m.group(1)): col
            for col in df.columns
            for m in [re.search(r'Timepoint_(\d+)', col)]
            if m
        }

        self.mask_dir   = mask_dir
        self.img_dir    = img_dir
        self.max_weeks  = max_weeks
        self.transform  = transform
        self.scaler     = MinMaxScaler()
        self.samples    = []

        # --- 3) build sample list ---
        all_masks = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))
        for mask_path in all_masks:
            fn = os.path.basename(mask_path)
            m = re.match(r'(PatientID_\d+)_Timepoint_(\d+)_tumorMask\.nii\.gz', fn)
            if not m: continue
            pid, tp = m.group(1), int(m.group(2))

            # corresponding T1c
            img_name = f'{pid}_Timepoint_{tp}_brain_t1c.nii.gz'
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path): continue

            # read time from clinical df
            col = self.tp_col_map.get(tp)
            if col is None: continue
            sid = pid if pid in df.index else pid.replace('PatientID_', '')
            raw = df.at[sid, col] if sid in df.index else np.nan
            if pd.isna(raw) or raw <= 0: continue
            days = float(raw)

            # find baseline
            pid_masks = [os.path.basename(p) for p in all_masks if p.startswith(os.path.join(mask_dir, pid))]
            tps       = [int(re.search(r'Timepoint_(\d+)', x).group(1)) for x in pid_masks]
            days_map  = {}
            for otp in tps:
                col2 = self.tp_col_map.get(otp)
                if not col2: continue
                val = df.at[sid, col2] if sid in df.index else np.nan
                if pd.notna(val) and val>0:
                    days_map[otp] = float(val)
            if not days_map: continue
            base_tp   = min(days_map, key=lambda k: days_map[k])
            base_days = days_map[base_tp]

            dt_norm = np.clip(((days - base_days)/7.0)/self.max_weeks, 0.0, 1.0)
            self.samples.append({
                'mask':    mask_path,
                'img':     img_path,
                'dt_norm': dt_norm
            })

        print(f"MUTimeConditionedDataset: built {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def sample_conditions(self, batch_size: int):
        """
        Randomly sample B conditioning volumes.
        Returns a FloatTensor (B, C_cond, D, H, W) on cuda.
        """
        idxs  = np.random.randint(0, len(self), size=batch_size)
        out   = []
        for i in idxs:
            s    = self.samples[i]
            mask = nib.load(s['mask']).get_fdata()[None].astype(np.float32)
            mask = (mask>0).astype(np.float32)
            t_emb = fourier_time_emb(s['dt_norm'])      # shape (2L,)
            # broadcast into volume
            D,H,W = mask.shape[1:]
            t_vol = np.stack([np.full((D,H,W), v, dtype=np.float32) for v in t_emb], axis=0)
            cond  = np.concatenate([mask, t_vol], axis=0)  # (1+2L, D, H, W)
            cond, _ = self.transform(cond, cond)          # just to resize & reorder
            out.append(torch.from_numpy(cond))
        return torch.stack(out, 0).cuda()

    def __getitem__(self, idx):
        s      = self.samples[idx]
        # mask channel
        mask   = nib.load(s['mask']).get_fdata()[None].astype(np.float32)
        mask   = (mask>0).astype(np.float32)
        # image channel
        img    = nib.load(s['img']).get_fdata()[None].astype(np.float32)
        img    = 2*(img - img.min())/(img.max()-img.min()) - 1.0

        # fourier time embedding
        t_emb  = fourier_time_emb(s['dt_norm'])   # (2L,)
        D,H,W  = mask.shape[1:]
        t_vol  = np.stack([np.full((D,H,W), v, dtype=np.float32) for v in t_emb], axis=0)

        cond   = np.concatenate([mask, t_vol], axis=0)  # (1+2L, D, H, W)
        if self.transform:
            cond, img = self.transform(cond, img)

        return {
            'input':  torch.from_numpy(cond),
            'target': torch.from_numpy(img)
        }

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # quick sanity check
    ds = MUTimeConditionedDataset(
        mask_dir   = 'dataset/brats2021/seg',
        img_dir    = 'dataset/brats2021/t1',
        clinical_xlsx = '/content/data/…xlsx',
        max_weeks  = 20.0
    )
    sample = ds[0]
    print("input shape:", sample['input'].shape)
    print("target shape:", sample['target'].shape)
    C_cond = sample['input'].shape[0]
    print(f"→ your UNet’s in_channels should be: {C_cond}")
