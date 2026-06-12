"""
Model Architecture: SegFormer (nvidia/mit-b2)
Binary Semantic Segmentation for Glacial Lake Detection
"""
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

def get_model(num_labels=2, pretrained_path=None):
    """
    Returns SegFormer mit-b2 model for binary segmentation.
    Args:
        num_labels: number of output classes (2: background + lake)
        pretrained_path: path to fine-tuned checkpoint, or None for base model
    """
    if pretrained_path:
        model = SegformerForSemanticSegmentation.from_pretrained(pretrained_path)
        processor = SegformerImageProcessor.from_pretrained(pretrained_path)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        processor = SegformerImageProcessor.from_pretrained(
            "nvidia/mit-b2",
            do_resize=True,
            size={"height": 512, "width": 512},
            do_normalize=True
        )
    return model, processor

if __name__ == "__main__":
    model, processor = get_model(pretrained_path="checkpoints/segformer_v2")
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    print(f"Architecture: SegFormer mit-b2")
    print(f"Input size: 512x512")
    print(f"Output classes: 2 (background, glacial_lake)")
