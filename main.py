import streamlit as st
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torchio as tio
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_erosion
import os

# Streamlit title and description
st.title("3D Brain Hemorrhage Detection and Visualization")
st.write("""
This app detects and visualizes brain hemorrhage using a U-Net model.
You can select a brain MRI scan (in NIfTI format) from the sample folder.
""")

# File selection from sample folder
sample_folder = 'sample'
files = [f for f in os.listdir(sample_folder) if f.endswith('.nii')]
selected_file = st.selectbox("Select a brain MRI scan", files)

# Load the model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1,
)

# Load the model weights with proper device mapping
model_path = 'unet_model_statedict_resnet34andimagenet_best.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the device for the model
model = model.to(device)

# Set the model to evaluation mode
model.eval()

# Load the selected image
image_path = os.path.join(sample_folder, selected_file)
image = nib.load(image_path)
image_data = image.get_fdata()

# Convert the image data to a 4D tensor (channels, x, y, z)
image_tensor = torch.tensor(image_data).unsqueeze(0)  # Add the channel dimension

# Resize the image data to 128x128x128 for faster computation
resize_transform = tio.Resize((128, 128, 128))
image_data_resized = resize_transform(image_tensor).squeeze().numpy()  # Remove the channel dimension after resizing

# Normalize and preprocess the image slices for the model
image_data_resized = (image_data_resized - np.min(image_data_resized)) / (np.max(image_data_resized) - np.min(image_data_resized))

# Initialize an empty array to store the prediction results
prediction_volume = np.zeros_like(image_data_resized)

# Process each slice individually
for i in range(image_data_resized.shape[2]):  # Loop through each slice along the axial plane
    slice_img = image_data_resized[:, :, i]
    slice_img = torch.tensor(slice_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add channel and batch dimensions

    # No need for padding as the image is already 128x128
    with torch.no_grad():
        slice_pred = model(slice_img)
        slice_pred = torch.sigmoid(slice_pred).cpu().numpy()

    # Store the predicted mask in the prediction volume
    prediction_volume[:, :, i] = slice_pred[0, 0]

# Create a brain mask based on intensity thresholding
brain_mask = image_data_resized > 0.1  # Adjust the threshold as needed

# Convert the stroke prediction to a binary mask
threshold = 0.9
stroke_mask = prediction_volume > threshold

# Generate borders by subtracting the eroded mask from the original mask
brain_border = brain_mask.astype(int) - binary_erosion(brain_mask).astype(int)
stroke_border = stroke_mask.astype(int) - binary_erosion(stroke_mask).astype(int)

# Extract the coordinates for the brain border and stroke border
brain_border_x, brain_border_y, brain_border_z = np.where(brain_border)
stroke_border_x, stroke_border_y, stroke_border_z = np.where(stroke_border)

# Create a 3D scatter plot for the brain border
brain_trace = go.Scatter3d(
    x=brain_border_x,
    y=brain_border_y,
    z=brain_border_z,
    mode='markers',
    marker=dict(
        size=1,
        color='blue',
        opacity=0.4
    ),
    name='Brain Border'
)

# Create a 3D scatter plot for the stroke border
stroke_trace = go.Scatter3d(
    x=stroke_border_x,
    y=stroke_border_y,
    z=stroke_border_z,
    mode='markers',
    marker=dict(
        size=1.5,
        color='red',
        opacity=0.6
    ),
    name='Stroke Border'
)

# Set up the layout of the plot with the legend at the bottom
layout = go.Layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,  # Position the legend below the plot
        xanchor="center",
        x=0.5
    ),
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True)
    ),
    margin=dict(l=0, r=0, b=0, t=0),
)

# Combine the traces and create the figure
fig = go.Figure(data=[brain_trace, stroke_trace], layout=layout)

# Display the figure
st.plotly_chart(fig)
