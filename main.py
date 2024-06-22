import streamlit as st
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import os

# Set locale to Thai
# (There is no direct i18n module in Streamlit for localization, manual translation is required)
# st.set_locale("th")

# Define the model architecture
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,  # We will load our own weights
    in_channels=1,
    classes=1,
)

# Load the model weights
model_path = './unet_model_statedict_resnet34andimagenet_best.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Set the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.eval()

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize to 256x256
    image = image / 255.0  # Normalize to [0,1]
    image = image[np.newaxis, np.newaxis, :, :]  # Add batch and channel dimensions
    image = torch.tensor(image, dtype=torch.float32)
    return image.to(device)

# Function to apply the model and get the segmentation mask
def get_segmentation_mask(image, threshold):
    with torch.no_grad():
        output = model(image)
        output = output.squeeze().cpu().numpy()
        output = (output > threshold).astype(np.uint8) * 255
    return output

# Function to list sample images in a directory
def list_sample_images(directory):
    valid_extensions = ['jpg', 'jpeg', 'png']
    return [f for f in os.listdir(directory) if any(f.lower().endswith(ext) for ext in valid_extensions)]

# Streamlit app
st.title("Brainstroke segmentation from CT-SCAN")

# Sidebar controls
st.sidebar.title("ตัวควบคุม")

# Sample image selection in sidebar
sample_images_directory = './sample'  # Directory containing sample images
sample_images = list_sample_images(sample_images_directory)
selected_sample_image = st.sidebar.selectbox("เลือกรูปภาพตัวอย่าง", ["ไม่มี"] + sample_images)

# Image upload in sidebar
uploaded_file = st.sidebar.file_uploader("หรืออัปโหลดรูปภาพ MRI สมอง", type=["jpg", "jpeg", "png"])

# Main content area
if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    # st.image(original_image, caption='รูปภาพที่อัปโหลด', use_column_width=True)
elif selected_sample_image != "ไม่มี":
    image_path = os.path.join(sample_images_directory, selected_sample_image)
    original_image = Image.open(image_path)
    # st.image(original_image, caption=f'รูปภาพตัวอย่างที่เลือก: {selected_sample_image}', use_column_width=True)

if 'original_image' in locals():
    # Preprocess the image
    input_image = preprocess_image(original_image)
    
    # Threshold slider in sidebar
    threshold = st.sidebar.slider('ค่าความเข้มของการแยกส่วน', 0.0, 1.0, 0.5, 0.01)
    
    # Get segmentation mask
    mask = get_segmentation_mask(input_image, threshold)
    
    # Display original and predicted images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption='รูปภาพต้นฉบับ', use_column_width=True)
    with col2:
        st.image(mask, caption='แมสการแยกส่วน', use_column_width=True)
