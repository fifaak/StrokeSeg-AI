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
import plotly.io as pio

# Sidebar with logo
st.sidebar.image('smte_logo.png', use_column_width=True)

# Sidebar for image selection and additional options
st.sidebar.title("3D Brain Stroke Visualization")
# Define the correct folder and file name
sample_folder = './sample'
selected_file = "CT_AVM.nii"  # Use the actual file name here

# Create a selectbox with only the specific file option
selected_file = st.sidebar.selectbox("เลือกตัวอย่าง", [selected_file])

# Construct the full path to the selected file
image_path = os.path.join(sample_folder, selected_file)



# Add additional code to use `image_path` as needed


# Slider for stroke threshold
stroke_threshold = st.sidebar.slider("เลือกความมั่นใจของการตรวจจับเส้นเลือด", min_value=0.0, max_value=1.0, value=0.9)

# Colormap selection for stroke visualization
colormap = st.sidebar.selectbox("เลือกสีเส้นเลือด", ["แดง", "ม่วง", "เหลือง", "แดงเหลือดหมู", "ฟ้าหมอง", "ฟ้า"])

# Define stroke color based on selected colormap
stroke_color_map = {
    "แดง": "#ff0000", 
    "ม่วง": "#440154",  # A color from the viridis colormap
    "เหลือง": "#f0f921",   # A color from the plasma colormap
    "แดงเหลือดหมู": "#6e003a",  # A color from the inferno colormap
    "ฟ้าหมอง": "#003f5c",
    "ฟ้า" : "#0000FF"   # Blue color for stroke
}

# Colormap selection for brain color
colormap_brain = st.sidebar.selectbox("เลือกสีสมอง", ["ฟ้า", "ม่วง", "เหลือง", "แดงเลือดหมู", "ฟ้าหมอง", "แดง"])

# Define brain color based on selected colormap
brain_color_map = {
    "แดง": "#ff0000", 
    "ม่วง": "#440154",  # A color from the viridis colormap
    "เหลือง": "#f0f921",   # A color from the plasma colormap
    "แดงเลือดหมู": "#6e003a",  # A color from the inferno colormap
    "ฟ้าหมอง": "#003f5c",
    "ฟ้า" : "#0000FF"   # Blue color for brain
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
model_path = './unet_model_statedict_resnet34andimagenet_best.pth'
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
default_confidence_threshold = 0.1
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
        opacity=0.5
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
            opacity=0.8
        ),
    )
    data_traces.append(stroke_trace)

# Set up the layout of the plot to maximize the 3D plot size and enable rotation
# Set up the layout of the plot to maximize the 3D plot size and enable rotation
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
        zaxis=dict(visible=True),
        aspectratio=dict(x=1, y=1.2, z=0.8),  # Adjust the aspect ratio (x, y, z)
        camera=dict(
            eye=dict(x=3, y=0, z=0)  # Set the initial side view along the x-axis
        )
    ),
    margin=dict(l=0, r=0, b=0, t=10),  # Adjust margins to ensure the plot occupies more space
    height=1000,  # Increase the height of the plot
    width=1000,   # Increase the width of the plot
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100}}],
                "label": "Rotate",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Stop",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 5, "t": 5},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "left",
        "y": 0,
        "yanchor": "top"
    }]
)

# Create the rotation frames to rotate around the z-axis for a side view
frames = [go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=2*np.cos(theta), y=2*np.sin(theta), z=0)))))
          for theta in np.linspace(0, 2*np.pi, 60)]

# Combine the traces and create the figure
fig = go.Figure(data=data_traces, layout=layout, frames=frames)


# Generate the image of the plot
img_bytes = pio.to_image(fig, format='png')

# Center the Plotly chart in the middle of the screen
st.markdown("""
    <style>
    .main .block-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .main .block-container > .element-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Display the plot in the main area
st.plotly_chart(fig, use_container_width=True)

# Move the download button to the sidebar
st.sidebar.download_button(
    label="Download Plot",
    data=img_bytes,
    file_name="brain_stroke_plot.png",
    mime="image/png"
)
