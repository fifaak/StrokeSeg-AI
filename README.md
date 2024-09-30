# StrokeSeg AI: Brain Stroke Segmentation and 3D Reconstruction

**StrokeSeg AI** is a deep learning project designed to segment brain strokes from CT scans using a U-Net architecture with a custom ResNet encoder. The project also includes 3D reconstruction from multiple segmented slices, enabling advanced visualization of hemorrhagic stroke regions.

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

| Model                    | Epochs | IoU   | Dice  | mAP   |
|--------------------------|--------|-------|-------|-------|
| U-Net (ResNet50)          | 200    | 0.743 | 0.792 | 0.748 |
| U-Net (ResNet101)         | 100    | 0.512 | 0.644 | 0.607 |
| U-Net (EfficientNet-B0)   | 140    | 0.401 | 0.458 | 0.564 |

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

![Segmentation Example](docs/segmentation_example.png)
![3D Reconstruction](docs/3d_reconstruction.png)

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

