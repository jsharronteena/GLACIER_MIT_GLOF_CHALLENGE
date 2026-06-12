"""
Inference script for glacial lake segmentation.
Usage: python inference.py --input_dir /path/to/images --output_dir /path/to/output
"""
import os, cv2, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from utils import postprocess

def run_inference(input_dir, output_dir, checkpoint="checkpoints/segformer_v2"):
    os.makedirs(output_dir, exist_ok=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device)
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    model.eval()

    images = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    print(f"Running inference on {len(images)} images...")

    for fname in tqdm(images):
        img  = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        enc  = processor(images=img, return_tensors="pt")
        pv   = enc["pixel_values"].to(device)

        with torch.no_grad():
            out = model(pixel_values=pv)

        logits = nn.functional.interpolate(
            out.logits, size=(512, 512),
            mode="bilinear", align_corners=False
        )
        pred  = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        final = postprocess(pred)
        cv2.imwrite(os.path.join(output_dir, fname), final * 255)

    print(f"Done! Masks saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/segformer_v2")
    args = parser.parse_args()
    run_inference(args.input_dir, args.output_dir, args.checkpoint)
