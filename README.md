# Explainable Hybrid Swin-UNet for Multi-Class Brain Tumor Classification

An explainable deep-learning application for classifying brain MRI images into No Tumor, Glioma, Meningioma, and Pituitary Tumor categories.

## Live Application

https://huggingface.co/spaces/Maciiii221/brain-tumor-ai

## Project Overview

This project implements an Explainable Hybrid Swin-UNet framework for multi-class brain tumor classification from MRI images.

The application combines model inference, class-wise probability estimation, Grad-CAM explainability, CAM-based localization, downloadable visual outputs, and automated PDF reporting.

## Supported Classes

- No Tumor
- Glioma
- Meningioma
- Pituitary Tumor

## Main Features

- Four-class brain MRI classification
- Hybrid Swin-UNet architecture
- Prediction confidence
- Class-wise probability scores
- Inference-time measurement
- Grad-CAM heatmap generation
- CAM-based localization visualization
- Downloadable PNG outputs
- Automated PDF report
- Streamlit interface
- Docker deployment
- Public Hugging Face Spaces hosting

## Technology Stack

- Python
- PyTorch
- Swin Transformer
- U-Net
- OpenCV
- NumPy
- Streamlit
- Grad-CAM
- ReportLab
- Docker
- Hugging Face Spaces

## Local Installation

Create a virtual environment:

```bash
py -m venv venv
```

Activate it on Windows:

```bash
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the trained checkpoint in the project root:

```text
best_model.pth
```

Run the application:

```bash
streamlit run app.py
```

## Model Checkpoint

The trained checkpoint is not stored in this GitHub repository because it exceeds GitHub's standard individual-file size limit.

The complete deployed model can be tested through the public Hugging Face application:

https://huggingface.co/spaces/Maciiii221/brain-tumor-ai

## Explainability

Grad-CAM is used to visualize class-discriminative image regions. The displayed bounding box is a CAM-based localization estimate rather than pixel-level tumor segmentation.

## Author

M Asif Chishti
