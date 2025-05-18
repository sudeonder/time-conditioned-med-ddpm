# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch

from torchvision.transforms import Compose, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model

from dataset import NiftiImageGenerator, MUTimeConditionedDataset
import torchio as tio

# GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------------------------------------------------------
# Command line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder',    type=str, required=True,
                    help="Path to the MU-Glioma-Post root folder (Patient subdirs)")
parser.add_argument('--clinicalfile',         type=str, required=False,
                    help="Path to MU Glioma Post clinical Excel file")
parser.add_argument('--maxweeks',             type=float, default=20.0,
                    help="Max number of weeks for normalizing Δt")
parser.add_argument('--input_size',           type=int, default=192,
                    help="Height and width of volumes (use 192 for BraTS4mod)")
parser.add_argument('--depth_size',           type=int, default=144,
                    help="Depth (number of slices) of volumes (use 144 for BraTS4mod)")
parser.add_argument('--num_channels',         type=int, default=64,
                    help="Base feature channels in U-Net")
parser.add_argument('--num_res_blocks',       type=int, default=1,
                    help="Number of residual blocks per scale in U-Net")
parser.add_argument('--train_lr',             type=float, default=1e-5,
                    help="Learning rate")
parser.add_argument('--batchsize',            type=int, default=1,
                    help="Batch size (use 1–2 for 3D volumes)")
parser.add_argument('--epochs',               type=int, default=50000,
                    help="Number of training iterations")
parser.add_argument('--timesteps',            type=int, default=250,
                    help="Number of diffusion timesteps")
parser.add_argument('--save_and_sample_every',type=int, default=1000,
                    help="Iteration interval for saving and sampling")
parser.add_argument('--with_condition',       action='store_true',
                    help="Enable mask+time conditioning")
parser.add_argument('-r', '--resume_weight',  type=str, default="",
                    help="Path to pretrained or checkpoint weights")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Resize transform: (240,240,155) → (192,192,144)
# -----------------------------------------------------------------------------
resizer = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad((240,240,155)),
    tio.Resize((192,192,144))
])

def cond_img_resize(cond, img):
    """
    cond: numpy array shape (2,240,240,155)
    img : numpy array shape (1,240,240,155)
    returns resized cond,img to (2,192,192,144) and (1,192,192,144)
    """
    c_img = tio.ScalarImage(tensor=cond)
    t_img = tio.ScalarImage(tensor=img)
    c_res = resizer(c_img).data.numpy()
    t_res = resizer(t_img).data.numpy()
    return c_res, t_res

# -----------------------------------------------------------------------------
# Single-channel transform (for no-condition mode)
# -----------------------------------------------------------------------------
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),        # (1,H,W,D)
    Lambda(lambda t: t.permute(0, 3, 1, 2))  # → (1,D,H,W)
])

# -----------------------------------------------------------------------------
# Dataset selection
# -----------------------------------------------------------------------------
if args.with_condition:
    assert args.clinicalfile, "Must provide --clinicalfile when using --with_condition"
    dataset = MUTimeConditionedDataset(
        root_dir=args.inputfolder,
        clinical_xlsx=args.clinicalfile,
        max_weeks=args.maxweeks,
        transform=cond_img_resize
    )
else:
    dataset = NiftiImageGenerator(
        args.inputfolder,
        input_size=args.input_size,
        depth_size=args.depth_size,
        transform=transform
    )

print(f"Loaded dataset with {len(dataset)} samples")

# -----------------------------------------------------------------------------
# Model & Diffusion setup
# -----------------------------------------------------------------------------
in_channels = 2 if args.with_condition else 1
out_channels = 1

model = create_model(
    args.input_size,
    args.num_channels,
    args.num_res_blocks,
    in_channels=in_channels,
    out_channels=out_channels
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size     = args.input_size,
    depth_size     = args.depth_size,
    timesteps      = args.timesteps,
    loss_type      = 'l1',
    with_condition = args.with_condition,
    channels       = out_channels
).cuda()

# Optionally load pretrained or checkpoint weights
if args.resume_weight:
    ckpt = torch.load(args.resume_weight, map_location='cuda')
    # load EMA if present
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
    train_lr                  = args.train_lr,
    train_num_steps           = args.epochs,
    gradient_accumulate_every = 2,
    ema_decay                 = 0.995,
    fp16                      = False,
    with_condition            = args.with_condition,
    save_and_sample_every     = args.save_and_sample_every,
)

# -----------------------------------------------------------------------------
# Run training
# -----------------------------------------------------------------------------
trainer.train()
