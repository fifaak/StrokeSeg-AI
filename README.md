# StrokeSeg AI: Brain Stroke Segmentation and 3D Reconstruction

**StrokeSeg AI** is a deep learning project designed to segment brain strokes from CT scans using a U-Net architecture with a custom ResNet encoder. The project also includes 3D reconstruction from multiple segmented slices, enabling advanced visualization of hemorrhagic stroke regions.
![Overview of Project's Architechture](docs/Overview.png)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [3D Reconstruction](#3d-reconstruction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

StrokeSeg AI leverages deep learning to segment and reconstruct brain strokes from CT scan images. This project aims to assist healthcare professionals by providing faster and more accurate stroke detection, specifically focusing on hemorrhagic strokes. The U-Net architecture, paired with a ResNet encoder, enhances the model's ability to detect stroke-affected regions with high precision.

## Key Features

- **U-Net with ResNet Encoder**: Combines the U-Net model with ResNet encoders (ResNet50/ResNet101) for improved feature extraction and segmentation accuracy.
- **3D Reconstruction**: Generates 3D visualizations from segmented slices for a comprehensive view of stroke regions.
- **Fast Inference**: Provides segmentation results in under 20 seconds, significantly faster than traditional diagnostic methods.
- **Web-based Application**: Deployable on cloud platforms, with a user-friendly web interface accessible by medical professionals.

## Model Architecture

The model utilizes a U-Net architecture with a pre-trained ResNet encoder, optimized for medical image segmentation. Key components include:

- **Encoder**: Uses pre-trained ResNet50/ResNet101 models for enhanced feature extraction.
- **Decoder**: U-Net decoder that upsamples encoded features to generate segmentation masks.
- **Transfer Learning**: Pre-trained on ImageNet to improve training efficiency and accuracy.

### Model Performance

The following table outlines the performance metrics for various models trained on the brain stroke segmentation task. Metrics include Intersection over Union (IoU), Dice coefficient, and mean Average Precision (mAP), which are key indicators of segmentation accuracy.

| Model                               | Epochs | Batch | Learning Rate | Threshold | IoU   | Dice  | mAP   |
|-------------------------------------|--------|-------|---------------|-----------|-------|-------|-------|
| Baseline (Densenet121/Unet)         | 40     | 8     | 0.0001        | 0.5       | 0.4738| 0.5331| 0.5720|
| Unet (Resnet34/Imagenet)            | 40     | 8     | 0.0001        | 0.5       | 0.3839| 0.4423| 0.4421|
| Unet (Resnet34/Imagenet)            | 100    | 8     | 0.0001        | 0.5       | 0.5059| 0.5591| 0.5874|
| Unet (Resnet50/Imagenet)            | 200    | 8     | 0.0001        | 0.5       | 0.7436| 0.7929| 0.7480|
| Unet (Resnet50/Imagenet)            | 200    | 32    | 0.0001        | 0.5       | 0.3911| 0.4580| 0.5033|
| Unet (Resnext101_32x8d/Imagenet)    | 100    | 8     | 0.0001        | 0.5       | 0.4017| 0.4580| 0.5638|
| Unet (Efficientnet-b0/Imagenet)     | 140    | 8     | 0.0001        | 0.5       | 0.4017| 0.4580| 0.5638|
| Deeplabv3 (Resnet101/Pretrained)    | 100    | 16    | 0.0001        | 0.5       | 0.5123| 0.6445| 0.6079|
| Deeplabv3 (Resnet50/Pretrained)     | 100    | 16    | 0.0001        | 0.5       | 0.4897| 0.6142| 0.5721|

### Best Model

- **U-Net (Resnet50/Imagenet)** with 200 epochs, a batch size of 8, and a learning rate of 0.0001 achieved the best performance with:
  - IoU: 0.7436
  - Dice: 0.7929
  - mAP: 0.7480


## 3D Reconstruction

StrokeSeg AI supports 3D reconstruction by stacking segmented layers from CT scans, offering a detailed volumetric representation of the brain. This feature provides medical professionals with an improved understanding of the stroke's location, size, and severity.

## Installation

To install and run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/fifaak/BrainStroke_Segmentation.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

Once the application is running, upload a CT scan to generate the stroke segmentation and corresponding 3D reconstruction. The output will be displayed in the web interface, allowing for easy interpretation by healthcare professionals.

## Results

The model shows promising results, achieving an IoU of 0.743 and a Dice coefficient of 0.792 using ResNet50 as the encoder. 3D reconstruction allows for enhanced visualization and understanding of stroke-affected areas.
###Segmentation Example
![Segmentation Example](docs/segment_2d.gif)
###3D Reconstruction after predicted
![3D Reconstruction](docs/3d_reconstruction.gif)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### Tags

- U-Net
- ResNet
- Stroke Segmentation
- Medical Imaging
- Deep Learning
- 3D Reconstruction
- Brain CT
- Hemorrhagic Stroke

