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
import io
import plotly.io as pio

# Sidebar with logo
st.sidebar.image('/Users/nimbuss/Downloads/3D-Brainreconstruct/BrainStroke_Segmentation/smte_logo.png', use_column_width=True)

# Sidebar for image selection and additional options
st.sidebar.title("3D Brain Stroke Visualization")

# Image selection options
sample_folder = 'sample'
files = [f for f in os.listdir(sample_folder) if f.endswith('.nii')]
selected_file = st.sidebar.selectbox("เลือกตัวอย่าง", files)
image_path = os.path.join(sample_folder, selected_file)

# Slider for stroke threshold
stroke_threshold = st.sidebar.slider("เลือกความมั่นใจของการตรวจจับเส้นเลือด", min_value=0.0, max_value=1.0, value=0.9)

# Colormap selection for stroke visualization
colormap = st.sidebar.selectbox("เลือกสีเส้นเลือด", ["Magma", "Viridis", "Plasma", "Inferno", "Cividis", "Blue"])

# Define stroke color based on selected colormap
stroke_color_map = {
    "Magma": "#ff0000", 
    "Viridis": "#440154",  # A color from the viridis colormap
    "Plasma": "#f0f921",   # A color from the plasma colormap
    "Inferno": "#6e003a",  # A color from the inferno colormap
    "Cividis": "#003f5c",
    "Blue" : "#0000FF"   # Blue color for stroke
}

# Colormap selection for brain color
colormap_brain = st.sidebar.selectbox("เลือกสีสมอง", ["Blue", "Viridis", "Plasma", "Inferno", "Cividis", "Magma"])

# Define brain color based on selected colormap
brain_color_map = {
    "Magma": "#ff0000", 
    "Viridis": "#440154",  # A color from the viridis colormap
    "Plasma": "#f0f921",   # A color from the plasma colormap
    "Inferno": "#6e003a",  # A color from the inferno colormap
    "Cividis": "#003f5c",
    "Blue" : "#0000FF"   # Blue color for brain
}

stroke_color = stroke_color_map.get(colormap, "#0000FF")  # Default to Blue if not found
brain_color = brain_color_map.get(colormap_brain, "#ff0000")  # Default to Red if not found

# Checkbox to toggle stroke border display
show_stroke_border = st.sidebar.checkbox("Show Stroke Border", value=True)

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

# Default brain mask threshold for full brain mask
default_confidence_threshold = 0.21
brain_mask = image_data_resized > default_confidence_threshold

# Convert the stroke prediction to a binary mask using the stroke threshold slider
stroke_mask = prediction_volume > stroke_threshold

# Generate borders by subtracting the eroded mask from the original mask
brain_border = brain_mask.astype(int) - binary_erosion(brain_mask).astype(int)
stroke_border = stroke_mask.astype(int) - binary_erosion(stroke_mask).astype(int)

# Extract the coordinates for the brain border and stroke border
brain_border_x, brain_border_y, brain_border_z = np.where(brain_border)
stroke_border_x, stroke_border_y, stroke_border_z = np.where(stroke_border)

# Create traces for the plot
data_traces = []

# Always add the brain border trace
brain_trace = go.Scatter3d(
    x=brain_border_x,
    y=brain_border_y,
    z=brain_border_z,
    mode='markers',
    marker=dict(
        size=1,
        color=brain_color,  # Use a single color directly for the brain border
        opacity=0.4
    ),
)
data_traces.append(brain_trace)

# Conditionally add the stroke border trace based on the checkbox
if show_stroke_border:
    stroke_trace = go.Scatter3d(
        x=stroke_border_x,
        y=stroke_border_y,
        z=stroke_border_z,
        mode='markers',
        marker=dict(
            size=1.5,
            color=stroke_color,  # Use a single color directly for the stroke border
            opacity=0.6
        ),
    )
    data_traces.append(stroke_trace)

# Set up the layout of the plot to maximize the 3D plot size
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
    margin=dict(l=10, r=10, b=10, t=10),  # Adjust margins to ensure the plot occupies more space
    height=1200,  # Increase the height of the plot
    width=1400    # Increase the width of the plot
)

# Combine the traces and create the figure
fig = go.Figure(data=data_traces, layout=layout)

# Generate the image of the plot
img_bytes = pio.to_image(fig, format='png')

# Create columns for the plot and the download button
col1, col2 = st.columns([4, 1])  # Adjust the ratio as needed

with col1:
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)  # Ensure the plot uses the full container width

with col2:
    # Add a download button to the right of the plot
    st.sidebar.download_button(
        label="ดาวน์โหลดผลลัพธ์",
        data=img_bytes,
        file_name='brain_stroke_plot.png',
        mime='image/png'
    )
