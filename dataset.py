#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import pandas as pd



def fourier_time_emb(dt_norm: float, L: int=6):
    freqs = 2.0 ** np.arange(L)
    angles = np.outer(freqs, 2*np.pi*dt_norm)  # shape (L,)
    sincos = np.concatenate([np.sin(angles), np.cos(angles)], axis=0)  # (2L,)
    return sincos  # shape (2L,)


# Create a TorchIO operator that takes a 2-chan cond or 1-chan img → resized volume
# instantiate once
resizer = tio.Resize((192, 192, 144))

def cond_img_resize(cond: np.ndarray, img: np.ndarray):
    """
    Resize and reorder so that output shapes are:
      cond → (2, 144, 192, 192)
      img  → (1, 144, 192, 192)
    """
    # cond, img come in as (C, X=240, Y=240, Z=155)
    # after resize → (C, 192, 192, 144)
    cond_r = resizer(tio.ScalarImage(tensor=cond)).data.numpy()
    img_r  = resizer(tio.ScalarImage(tensor=img)).data.numpy()
    # Now swap axes: (C, 192 (H), 192 (W), 144 (D)) → (C, D, H, W)
    cond_swapped = cond_r.transpose(0, 3, 1, 2)
    img_swapped  = img_r.transpose(0, 3, 1, 2)
    return cond_swapped, img_swapped

class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 3,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output

    def pair_file(self):
        input_files = sorted(glob(os.path.join(self.input_folder, '*')))
        target_files = sorted(glob(os.path.join(self.target_folder, '*')))
        pairs = []
        for input_file, target_file in zip(input_files, target_files):
            assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            pairs.append((input_file, target_file))
        return pairs

    def label2masks(self, masked_img):
        result_img = np.zeros(masked_img.shape + ( self.input_channel - 1,))
        result_img[masked_img==LabelEnum.BRAINAREA.value, 0] = 1
        result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                buff = input_img.copy()[..., ch]
                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
        return torch.cat(input_tensors, 0).cuda()

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
        input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
        input_img = self.resize_img(input_img) if not self.full_channel_mask else self.resize_img_4d(input_img)

        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)

        return {'input':input_img, 'target':target_img}
    

class MUTimeConditionedDataset(Dataset):
    """
    Flat-folder loader for MU-Glioma-Post with time-conditioning.
    Returns a dict with keys 'input' (mask+time) and 'target' (T1c).
    """
    def __init__(self,
                 mask_dir: str,
                 img_dir: str,
                 clinical_xlsx: str,
                 max_weeks: float,
                 transform=cond_img_resize):
        # 1) Clinical DataFrame
        xls = pd.ExcelFile(clinical_xlsx)
        sheet = xls.sheet_names[1]  # adjust if needed
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = df.columns.str.strip()
        df['Patient_ID'] = df['Patient_ID'].astype(str).str.strip()
        df.set_index('Patient_ID', inplace=True)
        self.clin_df = df

        # 2) Timepoint→column map
        self.tp_col_map = {
            int(m.group(1)): col
            for col in df.columns
            for m in [re.search(r'Timepoint_(\d+)', col)]
            if m
        }

        self.mask_dir = mask_dir
        self.img_dir  = img_dir
        self.max_weeks = max_weeks
        self.transform = transform

        # 3) Gather samples
        mask_paths = sorted(glob(os.path.join(mask_dir, '*.nii.gz')))
        self.samples = []
        for mask in mask_paths:
            fn = os.path.basename(mask)
            m = re.match(r'(PatientID_\d+)_Timepoint_(\d+)_tumorMask\.nii\.gz', fn)
            if not m: continue
            pid, tp = m.group(1), int(m.group(2))

            # corresponding image
            img_name = f"{pid}_Timepoint_{tp}_brain_t1c.nii.gz"
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue

            # get clinical day
            col = self.tp_col_map.get(tp)
            if col is None:
                continue
            sid = pid if pid in df.index else pid.replace('PatientID_', '')
            raw = df.at[sid, col] if sid in df.index else np.nan
            if pd.isna(raw) or raw <= 0:
                continue
            days = float(raw)

            # find baseline (across this pid’s masks)
            pid_masks = [os.path.basename(p) 
                         for p in mask_paths 
                         if os.path.basename(p).startswith(pid)]
            tps = [int(re.search(r'Timepoint_(\d+)', nm).group(1)) for nm in pid_masks]
            days_map = {}
            for otp in tps:
                c2 = self.tp_col_map.get(otp)
                if not c2: continue
                val = df.at[sid, c2] if sid in df.index else np.nan
                if pd.notna(val) and val > 0:
                    days_map[otp] = float(val)
            if not days_map:
                continue
            base_tp = min(days_map, key=days_map.get)
            base_days = days_map[base_tp]

            # normalized Δt
            dt_norm = np.clip(((days - base_days)/7.0)/self.max_weeks, 0.0, 1.0)

            self.samples.append({
                'mask':    mask,
                'img':     img_path,
                'dt_norm': dt_norm
            })

        print(f"MUTimeConditionedDataset: built {len(self.samples)} samples")
        
    def sample_conditions(self, batch_size: int):
        """
        Randomly sample `batch_size` conditioning inputs (mask+time)
        and return a CUDA tensor of shape (B, 2, D, H, W).
        """
        # 1) choose random indices
        idxs = np.random.randint(0, len(self), size=batch_size)
        conds = []
        for i in idxs:
            s = self.samples[i]
            # load & binarize mask
            mask = nib.load(s['mask']).get_fdata()[None].astype(np.float32)
            mask = (mask > 0).astype(np.float32)
            # time map
            tmap = np.full_like(mask, fill_value=s['dt_norm'])
            cond = np.concatenate([mask, tmap], axis=0)  # (2, H, W, D)
            # resize + reorder
            cond, _ = self.transform(cond, cond)       # we only need the first return
            conds.append(torch.from_numpy(cond))
        # stack and move to GPU
        return torch.stack(conds, dim=0).cuda()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # load & binarize mask
        mask = nib.load(s['mask']).get_fdata()[None].astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        # load & normalize image
        img = nib.load(s['img']).get_fdata()[None].astype(np.float32)
        img = 2*(img - img.min())/(img.max()-img.min()) - 1
        # L = 6
        ft = fourier_time_emb(s['dt_norm'], L=L)      # shape (2L,)
        # now tile to full volume:
        # mask: (1, D, H, W)
        D,H,W = mask.shape[1:]
        time_vol = np.zeros((2*L, D, H, W), dtype=np.float32)
        for i, val in enumerate(ft):
            time_vol[i] = val   # broadcast
        cond = np.concatenate([mask, time_vol], axis=0)  # now C = 1+2L
        # time map
        #tmap = np.full_like(mask, fill_value=s['dt_norm'])
        #cond = np.concatenate([mask, tmap], axis=0)

        # apply resize (if provided)
        if self.transform:
            cond, img = self.transform(cond, img)

        # return raw (C, D, H, W) so DataLoader yields (B, C, D, H, W)
        return {
              'input':  torch.from_numpy(cond),
              'target': torch.from_numpy(img)
          }
