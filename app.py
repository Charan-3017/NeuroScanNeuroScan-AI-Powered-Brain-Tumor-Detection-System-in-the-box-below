
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import sys
import os

# Add scripts directory to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
import preprocessing
import unet_model

# Constants
MODEL_INPUT_SHAPE = (256, 256)
THRESHOLD = 0.5

@st.cache_resource
def load_model():
    """Load the U-Net model. Creates a fresh instance if no weights found."""
    # Build model architecture
    model = unet_model.build_unet(input_shape=(MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1], 1))
    
    # Compile (needed for some saving/loading scenarios, or just good practice)
    model.compile(optimizer='adam', loss=unet_model.dice_loss, metrics=[unet_model.dice_coef])
    
    # Try to load weights if they exist (placeholder for future training step)
    weights_path = 'model_weights.h5'
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("Loaded trained weights.")
        except Exception as e:
            st.warning(f"Could not load weights: {e}. Using random weights.")
    else:
        st.info("No trained weights found. Using random weights for demonstration.")
        
    return model

def predict_tumor(model, image_array, demo_mode=False):
    """Run inference on a single image."""
    # Preprocess
    img_resized = preprocessing.resize_image(image_array, MODEL_INPUT_SHAPE)
    img_clahe = preprocessing.apply_clahe(img_resized)
    img_norm = preprocessing.normalize_to_01(img_clahe)
    
    if demo_mode:
        # Generate a synthetic "tumor" mask for demonstration
        # Create a Gaussian blob in a fixed or random spot
        rows, cols = MODEL_INPUT_SHAPE
        y, x = np.ogrid[:rows, :cols]
        center_y, center_x = 100, 150
        radius = 30
        mask_val = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2. * radius**2))
        pred_mask = mask_val # Soft mask
        return img_resized, img_clahe, pred_mask
    
    # Expand dims for batch: (1, 256, 256, 1)
    input_tensor = np.expand_dims(img_norm, axis=0) # (1, 256, 256)
    input_tensor = np.expand_dims(input_tensor, axis=-1) # (1, 256, 256, 1)
    
    # Predict
    prediction = model.predict(input_tensor)
    pred_mask = prediction[0, :, :, 0] # (256, 256)
    
    return img_resized, img_clahe, pred_mask

def create_heatmap_overlay(original_gray, pred_mask, threshold=0.5):
    """Create a heatmap overlay only where prediction is significant."""
    # 1. Create colored heatmap
    # Convert mask to uint8
    heatmap_uint8 = (pred_mask * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # 2. Convert original to RGB
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
    
    # 3. Create transparency mask based on prediction confidence
    # We only want to overlay color where confidence is > low threshold (e.g. 0.1)
    # Or purely proportional to confidence.
    
    # Create final image data container
    overlay = original_rgb.astype(np.float32)
    heatmap_float = heatmap_color.astype(np.float32)
    
    # Alpha blending: 
    # alpha = pred_mask * opacity_factor
    # out = src * (1 - alpha) + overlay * alpha
    
    opacity = 0.6
    # Expand pred_mask to 3 channels for broadcasting
    alpha = np.dstack([pred_mask] * 3) * opacity
    
    # Blend
    composite = original_rgb * (1.0 - alpha) + heatmap_float * alpha
    
    # Clip and convert back to uint8
    overlay_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
    
    # 4. Optional: Draw distinct contour for high confidence
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Thicker distinct contour (Cyan) for better visibility
    cv2.drawContours(overlay_uint8, contours, -1, (0, 255, 255), 3)
    
    return overlay_uint8

def calculate_metrics(pred_mask, threshold=0.5):
    """Calculate confidence score and area."""
    # Mask of predicted tumor
    tumor_region = pred_mask > threshold
    tumor_pixels = np.sum(tumor_region)
    
    # Confidence: Mean probability in the predicted tumor region
    if tumor_pixels > 0:
        confidence_score = np.mean(pred_mask[tumor_region])
    else:
        # If no tumor predicted, confidence is effectively 0 for "tumor presence" 
        # or we could say confidence in "no tumor" is high. 
        # Detailed logic varies, here we return max prob as a proxy or 0.
        confidence_score = np.max(pred_mask)
        
    return confidence_score, tumor_pixels

# --- UI Setup ---
st.set_page_config(page_title="NeuroScan", page_icon="🧠", layout="wide")

st.title("🧠 NeuroScan: Brain Tumor Detection")
st.markdown("""
Upload a brain MRI scan to detect potential tumors.
The app leverages a **U-Net** deep learning model to segment and highlight tumor regions.
""")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Choose an MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale
    image_np = np.array(image)
    
    # Check for trained weights to set default behavior
    weights_path = 'model_weights.h5'
    weights_exist = os.path.exists(weights_path)
    
    st.sidebar.header("Settings")
    # Default to Demo Mode if no weights found
    demo_default = not weights_exist
    
    demo_mode = st.sidebar.checkbox("Demo Mode (Simulate Tumor)", value=demo_default, help="Use a synthetic tumor mask to demonstrate visualization (useful if model is untrained).")
    
    if not weights_exist and not demo_mode:
        st.warning("⚠️ No trained model found and Demo Mode is OFF. Output will be random noise.")
    elif demo_mode:
        st.info("ℹ️ Demo Mode Active: Displaying synthetic tumor overlay.")
    
    st.sidebar.header("Processing")
    with st.spinner("Loading Model & Analyzing..."):
        model = load_model()
        
        # Inference
        img_resized, img_clahe, pred_mask = predict_tumor(model, image_np, demo_mode=demo_mode)
        
        # Metrics
        confidence, area = calculate_metrics(pred_mask, THRESHOLD)
        
        # Overlay
        overlay = create_heatmap_overlay(img_clahe, pred_mask, THRESHOLD)
    
    # Display Results
    st.success("Analysis Complete")
    
    # Metrics Layout
    met1, met2, met3 = st.columns(3)
    met1.metric("Tumor Detected", "Yes" if area > 0 else "No")
    met2.metric("Confidence Score", f"{confidence:.2%}")
    met3.metric("Est. Tumor Area", f"{int(area)} px")
    
    # Visuals
    scale_col1, scale_col2 = st.columns(2)
    
    with scale_col1:
        st.subheader("Original Scan (CLAHE)")
        # Show the processed input (CLAHE) because that's what the model sees
        st.image(img_clahe, caption="Processed Input", use_container_width=True, clamp=True, channels='GRAY')
        
    with scale_col2:
        st.subheader("Tumor Highlight")
        st.image(overlay, caption="Prediction Overlay", use_container_width=True, clamp=True, channels='RGB')
        
    with st.expander("See Raw Prediction Mask"):
        st.image(pred_mask, caption="Probability Heatmap", use_container_width=True, clamp=True)

else:
    st.info("Please upload an image to begin.")
