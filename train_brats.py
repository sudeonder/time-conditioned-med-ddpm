# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
import torchio as tio
from torchvision.transforms import Compose, Lambda
import numpy as np

from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from dataset import MUTimeConditionedDataset  # your updated dataset

# -----------------------------------------------------------------------------
# GPU setup
# -----------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-i',  '--seg_folder',   type=str, required=True,
                    help="Flat seg/ directory of *_tumorMask.nii.gz")
parser.add_argument('-t1', '--t1_folder',    type=str, required=True,
                    help="Flat t1/ directory of *_brain_t1c.nii.gz")
parser.add_argument('--clinicalfile',        type=str, required=True,
                    help="Path to MU-Glioma-Post clinical Excel")
parser.add_argument('--maxweeks',            type=float, default=20.0,
                    help="Max weeks for normalizing Δt")
parser.add_argument('--input_size',          type=int, default=192,
                    help="Height & width for training")
parser.add_argument('--depth_size',          type=int, default=144,
                    help="Depth (slices) for training")
parser.add_argument('--num_channels',        type=int, default=64,
                    help="Base UNet channels")
parser.add_argument('--num_res_blocks',      type=int, default=2,
                    help="Residual blocks per scale")
parser.add_argument('--batchsize',           type=int, default=1,
                    help="Batch size")
parser.add_argument('--epochs',              type=int, default=100000,
                    help="Number of training iterations")
parser.add_argument('--timesteps',           type=int, default=250,
                    help="Diffusion timesteps")
parser.add_argument('--save_and_sample_every', type=int, default=1000,
                    help="Checkpoint & sample interval")
parser.add_argument('--with_condition',      action='store_true',
                    help="Enable mask + time conditioning")
parser.add_argument('-r', '--resume_weight', type=str, default="model/model_brats.pt",
                    help="Path to pretrained weights")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Define resize transform to match (192,192,144)
# -----------------------------------------------------------------------------
resizer = tio.Resize((args.input_size, args.input_size, args.depth_size))
def cond_img_resize(cond: np.ndarray, img: np.ndarray):
    # cond: (2,240,240,155), img: (1,240,240,155)
    cond_tio = tio.ScalarImage(tensor=cond)
    img_tio  = tio.ScalarImage(tensor=img)
    cond_r   = resizer(cond_tio).data.numpy()
    img_r    = resizer(img_tio).data.numpy()
    return cond_r, img_r

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
if args.with_condition:
    dataset = MUTimeConditionedDataset(
        mask_dir      = args.seg_folder,
        img_dir       = args.t1_folder,
        clinical_xlsx = args.clinicalfile,
        max_weeks     = args.maxweeks,
        transform     = cond_img_resize
    )
    print(f"Loaded dataset with {len(dataset)} samples")
else:
    raise RuntimeError("train_brats.py must be run with --with_condition for this project")

# -----------------------------------------------------------------------------
# Model & Diffusion setup
# -----------------------------------------------------------------------------
in_channels  = 2  # [mask, time_map]
out_channels = 1  # single T1-c output

model = create_model(
    args.input_size,
    args.num_channels,
    args.num_res_blocks,
    in_channels = in_channels,
    out_channels = out_channels
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size      = args.input_size,
    depth_size      = args.depth_size,
    timesteps       = args.timesteps,
    loss_type       = 'l1',
    with_condition  = args.with_condition,
    channels        = out_channels
).cuda()

# optionally load pretrained / checkpoint
if args.resume_weight:
    ckpt = torch.load(args.resume_weight, map_location='cuda')
    if 'ema' in ckpt:
        diffusion.load_state_dict(ckpt['ema'])
    else:
        diffusion.load_state_dict(ckpt)
    print(f"Loaded weights from {args.resume_weight}")

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
trainer = Trainer(
    diffusion,
    dataset,
    image_size                = args.input_size,
    depth_size                = args.depth_size,
    train_batch_size          = args.batchsize,
    train_lr                  = 1e-5,
    train_num_steps           = args.epochs,
    gradient_accumulate_every = 2,
    ema_decay                 = 0.995,
    fp16                      = False,
    with_condition            = args.with_condition,
    save_and_sample_every     = args.save_and_sample_every,
    results_folder            = './results_brats'
)

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
trainer.train()
