# GLOFeagles '26 Challenge Submission

## Team Information

**Team Name:** GlacierMIT

### Team Members
1. J. Sharron Teena
2. D. Hrishikesh

---

## Challenge Overview

This project focuses on automated glacial lake segmentation from satellite imagery for the GLOFEagles '26 Challenge. Accurate detection and delineation of glacial lakes are essential for monitoring environmental change and assessing Glacial Lake Outburst Flood (GLOF) risks.

Our solution utilizes a SegFormer MIT-B2 transformer-based semantic segmentation model combined with pseudo-labeling and self-training techniques. By leveraging both labeled and high-confidence unlabeled data, the model achieved strong segmentation performance while maintaining reproducibility and scalability across the complete challenge dataset.

---

## Methodology

### Model Architecture

- Backbone: SegFormer MIT-B2
- Framework: PyTorch
- Library: Hugging Face Transformers
- Task: Binary Semantic Segmentation
- Class 0: Background
- Class 1: Glacial Lake

### Image Processing

- Image resizing to 512 × 512 pixels
- Normalization using SegFormer image processor
- Data augmentation:
- Horizontal Flip
- Vertical Flip


### Training Strategy

#### Phase 1: Initial Supervised Training

- Labeled Images: 60
- Train/Validation Split: 80/20
- Optimizer: AdamW
- Learning Rate: 6e-5
- Epochs: 50

**Initial Performance**

| Metric | Score |
|----------|----------|
| Dice Score | 0.6611 |
| IoU | 0.5090 |

#### Phase 2: Pseudo-Labeling and Self-Training

To improve performance, the trained model generated pseudo-labels for unlabeled images.

- Unlabeled Images: 515
- High-Confidence Pseudo-Labels Generated: 387
- Combined Training Dataset: 447 Images

The model was fine-tuned using:

- AdamW Optimizer
- Learning Rate: 2e-5
- OneCycleLR Scheduler
- Early Stopping

### Key Design Choices

#### SegFormer over U-Net/DeepLab

The hierarchical transformer architecture captures both local texture information and global contextual information, making it highly suitable for glacial lakes ranging from small dark ponds to large cyan or turbid water bodies across diverse satellite imagery.

#### Pseudo-Labeling

Training with only 60 labeled images was insufficient to generalize across the diverse glacial lake appearances present in the dataset. Pseudo-labeling expanded the effective training dataset by approximately 7.4× without requiring additional manual annotation.

#### Fine-Tuning from Checkpoint

Instead of training from scratch, the model was fine-tuned from the Phase 1 checkpoint. This preserved previously learned lake features and reduced the risk of catastrophic forgetting.

#### Confidence Threshold

A confidence threshold of 0.85 was applied during pseudo-label selection to ensure only high-quality masks were incorporated into training.

---

## Training Highlights

- Generated 387 high-confidence pseudo-labels from 515 unlabeled images.
- Expanded the training dataset from 60 to 447 images.
- Fine-tuned SegFormer MIT-B2 using pseudo-labeled data.
- Achieved a peak validation Dice score of 0.9348 during training.
- Generated segmentation masks for all 575 challenge images.

---

## Final Evaluation Results

The final model was evaluated on the 60 labeled competition images containing ground-truth segmentation masks.

| Metric | Score |
|----------|----------|
| mIoU | **0.7884** |
| Dice Score | **0.8817** |
| F1 Score | **0.8817** |
| Precision | **0.9042** |
| Recall | **0.8603** |
| Accuracy | **0.9938** |
| Cohen's Kappa | **0.8785** |

### Performance Summary

The proposed SegFormer-based framework achieved strong segmentation performance across diverse glacial lake appearances, including clear-water, turbid, snow-adjacent, and partially obscured lakes.

These results demonstrate robust segmentation performance despite severe class imbalance, where glacial lakes occupy less than 5% of pixels on average.

---

## Prediction Statistics

| Statistic | Value |
|------------|--------|
| Total Images Predicted | 575 |
| Images Containing Lakes | 461 (80.17%) |
| Images Without Lakes | 114 (19.83%) |
| Average Lake Coverage | ~2.1% |

---


## Video Demonstration

https://youtu.be/nfflP24Gc3c

##Trained Model file link: 
DOWNLOAD LINK: 
https://drive.google.com/drive/folders/1_805-7BXigSHiwNy7gMrz3apmHXhBlYw?usp=sharing

---

## Acknowledgements

We thank the organizers of the GLOFEagles '26 Challenge for providing the dataset, benchmark framework, and evaluation protocol for glacial lake segmentation research.

---

## Contact

7358344496
jsharronteena@gmail.com
J. Sharron Teena
