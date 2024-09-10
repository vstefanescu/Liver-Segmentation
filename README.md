# Liver Segmentation

This repository contains an application for liver segmentation using neural networks, specifically the U-Net architecture. The goal of this project is to automate the process of liver segmentation from CT scan images using deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project applies the U-Net architecture to segment livers from CT scans, leveraging the **LITS17 dataset**. The application includes:
- Data loading and preprocessing for `.nii` medical images.
- U-Net model implementation for image segmentation.
- Training, validation, and testing on liver CT scans.
- Visualization of the original CT scan, ground truth, and predicted segmentation.

## Dataset

We use the **LITS17 (Liver Tumor Segmentation Challenge)** dataset for training and evaluation. The dataset includes:
- CT volumes of the liver (stored in `.nii` format).
- Corresponding segmentation masks (also in `.nii` format).

The dataset is structured into training, validation, and test sets as follows:

```text
dataset/
│
├── train/
│   ├── volume/
│   └── segmentation/
├── validation/
│   ├── volume/
│   └── segmentation/
└── test/
    ├── volume/
    └── segmentation/
