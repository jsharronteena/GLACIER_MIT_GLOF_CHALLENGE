"""
Training script for glacial lake segmentation using SegFormer.
Usage: python train.py --images_dir /path/to/images --masks_dir /path/to/masks
"""
import os, cv2, shutil, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
from tqdm import tqdm
from utils import dice_score, iou_score

class LakeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, augment=False):
        self.files     = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.processor = processor
        self.augment   = augment

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir,  fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        if self.augment:
            if np.random.rand() > 0.5: img, mask = img.transpose(2), mask.transpose(2)
            if np.random.rand() > 0.5: img, mask = img.transpose(1), mask.transpose(1)
            if np.random.rand() > 0.5: img, mask = img.transpose(0), mask.transpose(0)
        mask_np = (np.array(mask) > 127).astype(np.int64)
        enc     = self.processor(images=img, return_tensors="pt")
        return enc["pixel_values"].squeeze(), torch.tensor(mask_np)

def train(images_dir, masks_dir, model_dir="checkpoints/segformer_v2",
          epochs=30, batch_size=8, lr=2e-5):
    processor = SegformerImageProcessor.from_pretrained("checkpoints/segformer_v1")
    model     = SegformerForSemanticSegmentation.from_pretrained("checkpoints/segformer_v1")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    all_files = sorted(os.listdir(images_dir))
    np.random.seed(42); np.random.shuffle(all_files)
    split     = int(0.85 * len(all_files))

    for s, files in [("train", all_files[:split]), ("val", all_files[split:])]:
        for d in ["Images","Masks"]: os.makedirs(f"/tmp/split/{s}/{d}", exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(images_dir, f), f"/tmp/split/{s}/Images/{f}")
            shutil.copy(os.path.join(masks_dir,  f), f"/tmp/split/{s}/Masks/{f}")

    train_loader = DataLoader(
        LakeDataset(f"/tmp/split/train/Images", f"/tmp/split/train/Masks",
                    processor, augment=True), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(
        LakeDataset(f"/tmp/split/val/Images",   f"/tmp/split/val/Masks",
                    processor, augment=False), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
        epochs=epochs, pct_start=0.1)

    best_dice, patience = 0.0, 0
    for epoch in range(epochs):
        model.train()
        for pv, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            pv, masks = pv.to(device), masks.to(device)
            loss = model(pixel_values=pv, labels=masks).loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        model.eval(); vd = 0
        with torch.no_grad():
            for pv, masks in val_loader:
                pv, masks = pv.to(device), masks.to(device)
                out    = model(pixel_values=pv, labels=masks)
                logits = nn.functional.interpolate(out.logits, size=masks.shape[-2:],
                                                   mode="bilinear", align_corners=False)
                vd    += dice_score(logits.argmax(dim=1), masks).item()
        vd /= len(val_loader)
        print(f"Epoch {epoch+1:02d} | Dice: {vd:.4f}")
        if vd > best_dice:
            best_dice = vd; patience = 0
            model.save_pretrained(model_dir); processor.save_pretrained(model_dir)
            print(f"  Saved (Dice: {best_dice:.4f})")
        else:
            patience += 1
            if patience >= 8: print("Early stopping"); break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir",  required=True)
    parser.add_argument("--model_dir",  default="checkpoints/segformer_v2")
    parser.add_argument("--epochs",     type=int, default=30)
    args = parser.parse_args()
    train(args.images_dir, args.masks_dir, args.model_dir, args.epochs)
