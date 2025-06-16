# Add these optimizations to your existing app.py

import streamlit as st
import cv2
import numpy as np
import gc
from typing import Optional

# Page configuration - add this at the very top after imports
st.set_page_config(
    page_title="Image Restoration App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Performance optimizations for cloud deployment
@st.cache_data
def load_and_process_image(uploaded_file):
    """Cache image loading to improve performance."""
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

@st.cache_data
def validate_image_size(image: np.ndarray, max_dimension: int = 2048) -> bool:
    """Validate image size for cloud deployment limits."""
    h, w = image.shape[:2]
    return max(h, w) <= max_dimension

def memory_cleanup():
    """Force garbage collection for memory management."""
    gc.collect()

def display_image_warning(image: np.ndarray):
    """Display warnings for large images."""
    h, w = image.shape[:2]
    size_mb = (h * w * 3) / (1024 * 1024)  # Rough size calculation
    
    if size_mb > 25:
        st.warning(f"⚠️ Large image detected ({size_mb:.1f}MB). Processing may be slow.")
    elif max(h, w) > 2048:
        st.warning("⚠️ Image is very large. Consider resizing for better performance.")

def resize_large_image(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
    """Resize large images to improve performance."""
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        if h > w:
            new_h, new_w = max_dimension, int(w * max_dimension / h)
        else:
            new_h, new_w = int(h * max_dimension / w), max_dimension
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Enhanced error handling wrapper
def safe_process_image(func, *args, **kwargs):
    """Wrapper for safe image processing with error handling."""
    try:
        with st.spinner("Processing image..."):
            result = func(*args, **kwargs)
            memory_cleanup()
            return result
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        st.info("💡 Try uploading a smaller image or adjusting the parameters.")
        return None

# Add progress indicators for better UX
def show_processing_progress():
    """Show processing progress for better user experience."""
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
    progress_bar.empty()

# Cloud-specific UI improvements
def add_performance_info():
    """Add performance information for users."""
    with st.expander("ℹ️ Performance Tips"):
        st.markdown("""
        **For best performance:**
        - Use images smaller than 2048x2048 pixels
        - JPEG format typically loads faster than PNG
        - Smaller brush sizes process faster in inpainting mode
        - Close other browser tabs if the app feels slow
        """)

# Add to your main() function
def enhanced_main():
    """Enhanced main function with cloud optimizations."""
    
    # Add performance info
    add_performance_info()
    
    # Your existing sidebar setup code here...
    
    if uploaded_file is not None:
        # Validate file size first
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
            st.error("File too large! Please upload an image smaller than 50MB.")
            return
        
        # Load image with caching
        original_image = load_and_process_image(uploaded_file)
        
        if original_image is None:
            return
        
        # Display image warnings
        display_image_warning(original_image)
        
        # Offer to resize large images
        if not validate_image_size(original_image):
            if st.button("🔄 Resize image for better performance"):
                original_image = resize_large_image(original_image)
                st.success("Image resized successfully!")
        
        # Your existing processing code here...
        # Just wrap heavy operations with safe_process_image()

# Example of wrapping existing functions
def enhanced_handle_blur_filters(image, option, processor, ui):
    """Enhanced blur filter handling with optimizations."""
    controls = ui.create_filter_controls(option)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    with col2:
        st.subheader(f"Result - {option}")
        
        # Use safe processing wrapper
        if option == 'Median Blur':
            processed = safe_process_image(
                processor.apply_median_blur, 
                image, 
                controls['kernel_size']
            )
        else:
            processed = safe_process_image(
                processor.apply_bilateral_filter,
                image, 
                controls['diameter'],
                controls['sigma_color'], 
                controls['sigma_space']
            )
        
        if processed is not None:
            result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(result_rgb)
            
            # Download functionality
            result_pil = Image.fromarray(result_rgb)
            st.sidebar.markdown("---")
            st.sidebar.markdown(
                ui.create_download_link(
                    result_pil, 
                    f'{option.lower().replace(" ", "_")}.jpg',
                    f'📥 Download {option} Result'
                ),
                unsafe_allow_html=True
            )

# Add deployment status indicator
def show_deployment_info():
    """Show deployment information."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Live Demo**")
    st.sidebar.caption("Running on Streamlit Cloud")
    
    # Add GitHub link
    st.sidebar.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)]"
        "(https://github.com/yourusername/image-restoration-app)"
    )

# Call this in your main function
show_deployment_info()