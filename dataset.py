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
    

## MU-Glioma-Post dataset class 
class MUTimeConditionedDataset(Dataset):
    def __init__(self, root_dir, clinical_xlsx, max_weeks, transform=None):
        self.root, self.transform, self.max_weeks = root_dir, transform, max_weeks
        df = pd.read_excel(clinical_xlsx, sheet_name='MU Glioma Post')
        self.clin_df = df.set_index('Patient_ID')
        # map timepoint→column
        self.tp_col_map = {}
        for c in self.clin_df.columns:
            m = re.search(r'\( *Timepoint_(\d+) *\)', c)
            if m: self.tp_col_map[int(m.group(1))] = c
        self.samples = []
        for pid in sorted(os.listdir(self.root)):
            pdir = os.path.join(self.root, pid)
            if not os.path.isdir(pdir): continue
            # available timepoints on disk
            tps = sorted(int(m.group(1)) for m in
                         (re.match(r'Timepoint_(\d+)',d) for d in os.listdir(pdir)) if m)
            if not tps: continue
            # days‐map from clinical
            days_map = {}
            for tp in tps:
                col = self.tp_col_map.get(tp)
                raw = self.clin_df.at[pid, col] if col else np.nan
                if pd.notna(raw): days_map[tp] = float(raw)
            if not days_map: continue
            base_tp = min(days_map, key=lambda k: days_map[k])
            base_days = days_map[base_tp]
            base_folder = f"Timepoint_{base_tp}"
            base_mask = os.path.join(pdir, base_folder,
                                     f"{pid}_{base_folder}_tumorMask.nii.gz")
            if not os.path.exists(base_mask): continue
            for tp in tps:
                if tp not in days_map: continue
                dt_norm = np.clip(((days_map[tp]-base_days)/7.0)/self.max_weeks, 0,1)
                img_path = os.path.join(pdir, f"Timepoint_{tp}",
                                        f"{pid}_Timepoint_{tp}_brain_t1c.nii.gz")
                if os.path.exists(img_path):
                    self.samples.append({'mask0':base_mask,
                                         'img':img_path,
                                         'dt_norm':dt_norm})
        print(f"MUTimeConditionedDataset: built {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        mask0 = nib.load(s['mask0']).get_fdata().astype(np.float32)[None]
        mask0 = (mask0>0).astype(np.float32)
        img   = nib.load(s['img']).get_fdata().astype(np.float32)[None]
        img   = 2*(img-img.min())/(img.max()-img.min())-1
        tmap  = np.full_like(mask0, fill_value=s['dt_norm'])
        cond  = np.concatenate([mask0, tmap], axis=0)
        if self.transform:
            cond, img = self.transform(cond, img)
        return cond, img

