# 🫁 MedVision AI

> AI-powered chest X-ray analysis with explainable deep learning.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Computer Vision](https://img.shields.io/badge/Computer-Vision-blue)
![GradCAM](https://img.shields.io/badge/Explainability-GradCAM-cyan)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)

---

## Overview

MedVision AI is an interactive biomedical computer vision application that analyzes chest X-rays using a deep learning model built with PyTorch.

The application allows users to upload a chest X-ray and receive:

- Pneumonia-class probability
- Interactive Grad-CAM attention visualization
- Adjustable decision threshold
- Model performance metrics
- Downloadable research report
- Session analysis history

This project demonstrates an end-to-end AI workflow from data preprocessing through deployment.

---

## Features

- Deep learning inference using ResNet-18
- Transfer learning
- Explainable AI with Grad-CAM
- Interactive Streamlit dashboard
- Patient-level train / validation / test split
- Weighted loss for class imbalance
- ROC curve evaluation
- Research report generation
- Analysis history tracking

---

## Model

Architecture:

- ResNet-18
- PyTorch
- Binary classification

Task:

No Finding vs Pneumonia

---

## Dataset

NIH ChestX-ray14 Dataset

This repository does **not** include the dataset due to licensing and file size.

---

## Results

Held-out Test Set

| Metric | Value |
|--------|-------:|
| ROC-AUC | **0.7744** |
| Sensitivity | **76.3%** |
| Specificity | **65.1%** |

---

## Technology Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Pillow
- Grad-CAM

---

## Disclaimer

This project is an educational research prototype.

It is **not** a medical device and must **not** be used for diagnosis or clinical decision-making.

---

## Future Work

- Vision Transformers
- Multi-label pathology classification
- DICOM support
- Batch inference
- Automatic report generation
- Cloud deployment
- Confidence calibration
- Faster inference

---

## Author

Leya Glazer

Biomedical Engineering + Computer Science

University of Texas at Dallas
