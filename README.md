# Moved Object Detection Pipeline

## Description
This repository contains a robust, end-to-end object detection pipeline designed to identify moved objects in surveillance footage (VIRAT dataset). Unlike standard detection approaches, this project implements a Feature-Difference Injection strategy: using PyTorch forward hooks to extract semantic features from a frozen ResNet-50 backbone and injecting the difference tensor directly into a DETR Transformer Encoder.

## Data Preparation
The project utilizes a modified subset of the VIRAT Video Dataset. The dataset can be found [here](https://drive.google.com/file/d/11SQy2jB8_er9GSdQIMVBQca6Rci_b7tr/view?usp=sharing). Ensure your data files are placed in the `data/` directory as shown in the tree structure below.

## Directory Structure
Ensure your project directory is organized as follows:

```bash
.
├── code
│   ├── dataload.py
│   ├── eval.py
│   ├── model.py
│   ├── plots             
│   └── train.py
├── data
│   ├── base
│   └── matched_annotations
├── report.pdf
└── README.md
```

## Installation & Environment
Ensure you have a Python environment set up with the necessary dependencies. You can install them using pip:
```bash
pip install torch torchvision torchmetrics transformers opencv-python numpy matplotlib tqdm
```
## Usage
### Training the Model
To train the model using the "Heads-Only" configuration (Experiment A - Best Performing), run the training script from the root directory:
```bash
python code/train.py
```
**Expected Duration:** ~2 hours (on NVIDIA A100 GPU)<br>
**Outputs:** Console logs will display Training/Validation loss per epoch.<br>
    Loss curves and metrics are saved to `code/plots/.`

For training the configuration for Experiment B, make modification in line 83 of `train.py`.

## Report
Find the report can be found in the `report.pdf` file. 
