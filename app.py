"""
Interactive Image Restoration Application

A comprehensive Streamlit application for image restoration using various
computer vision techniques including median blur, bilateral filtering,
and advanced inpainting algorithms.

Author: [Adam Rangwala]
Date: [06/16/2025]"""

import streamlit as st
import cv2
import numpy as np
import io
import base64
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles all image processing operations."""
    
    @staticmethod
    def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply median blur to reduce noise while preserving edges."""
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, diameter: int, 
                             sigma_color: int, sigma_space: int) -> np.ndarray:
        """Apply bilateral filter for edge-preserving smoothing."""
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    @staticmethod
    def apply_inpainting(image: np.ndarray, mask: np.ndarray, 
                        method: str = 'telea') -> np.ndarray:
        """Apply inpainting to restore damaged image regions."""
        if method.lower() == 'telea':
            flag = cv2.INPAINT_TELEA
        else:
            flag = cv2.INPAINT_NS
        
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=flag)


class UIComponents:
    """Handles UI component generation and management."""
    
    @staticmethod
    def create_download_link(img: Image.Image, filename: str, text: str) -> str:
        """Generate a download link for processed images."""
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    
    @staticmethod
    def setup_sidebar() -> dict:
        """Configure sidebar components and return user selections."""
        st.sidebar.title('🖼️ Image Restoration Toolkit')
        st.sidebar.markdown("---")
        
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Image to Restore:",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        return {"uploaded_file": uploaded_file}
    
    @staticmethod
    def create_filter_controls(option: str) -> dict:
        """Create appropriate controls based on selected filter option."""
        controls = {}
        
        if option == 'Median Blur':
            controls['kernel_size'] = st.sidebar.slider(
                "Kernel Size:", 3, 15, 5, 2,
                help="Larger values create stronger blur effect"
            )
        
        elif option == 'Bilateral Blur':
            controls['diameter'] = st.sidebar.slider(
                "Diameter:", 1, 50, 20,
                help="Diameter of pixel neighborhood"
            )
            controls['sigma_color'] = st.sidebar.slider(
                "Sigma Color:", 0, 250, 200, 10,
                help="Filter sigma in the color space"
            )
            controls['sigma_space'] = st.sidebar.slider(
                "Sigma Space:", 0, 250, 100, 10,
                help="Filter sigma in the coordinate space"
            )
        
        return controls


def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title="Image Restoration App",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🖼️ Interactive Image Restoration")
    st.markdown("""
    Transform your images using advanced computer vision techniques. 
    Upload an image and choose from various restoration methods.
    """)
    
    # Initialize components
    processor = ImageProcessor()
    ui = UIComponents()
    
    # Setup sidebar
    sidebar_data = ui.setup_sidebar()
    uploaded_file = sidebar_data["uploaded_file"]
    
    if uploaded_file is None:
        # Display welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👆 Please upload an image to get started!")
            st.markdown("""
            ### Available Features:
            - **Median Blur**: Remove noise while preserving edges
            - **Bilateral Filter**: Edge-preserving smoothing
            - **Image Inpainting**: Remove unwanted objects or restore damaged areas
            """)
        return
    
    try:
        # Process uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_image is None:
            st.error("Failed to load image. Please try a different file.")
            return
        
        # Display original image info
        h, w, c = original_image.shape
        st.sidebar.markdown(f"""
        **Image Info:**
        - Dimensions: {w} × {h}
        - Channels: {c}
        - Size: {uploaded_file.size / 1024:.1f} KB
        """)
        
        # Filter selection
        st.sidebar.markdown("---")
        option = st.sidebar.selectbox(
            '🎛️ Choose Restoration Method:',
            ('None', 'Median Blur', 'Bilateral Blur', 'Image Inpainting'),
            help="Select the image processing technique to apply"
        )
        
        # Process based on selection
        if option == 'None':
            st.subheader("Original Image")
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        elif option in ['Median Blur', 'Bilateral Blur']:
            handle_blur_filters(original_image, option, processor, ui)
        
        elif option == 'Image Inpainting':
            handle_inpainting(original_image, uploaded_file, processor, ui)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"An error occurred while processing the image: {str(e)}")


def handle_blur_filters(image: np.ndarray, option: str, 
                       processor: ImageProcessor, ui: UIComponents):
    """Handle median blur and bilateral filter operations."""
    controls = ui.create_filter_controls(option)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    with col2:
        st.subheader(f"Result - {option}")
        
        if option == 'Median Blur':
            processed = processor.apply_median_blur(image, controls['kernel_size'])
        else:  # Bilateral Blur
            processed = processor.apply_bilateral_filter(
                image, controls['diameter'],
                controls['sigma_color'], controls['sigma_space']
            )
        
        result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        st.image(result_rgb)
        
        # Download link
        result_pil = Image.fromarray(result_rgb)
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            ui.create_download_link(result_pil, f'{option.lower().replace(" ", "_")}.jpg', 
                                  f'📥 Download {option} Result'),
            unsafe_allow_html=True
        )


def handle_inpainting(image: np.ndarray, uploaded_file, 
                     processor: ImageProcessor, ui: UIComponents):
    """Handle image inpainting operations."""
    st.subheader("🎨 Interactive Inpainting")
    st.markdown("Draw on the image to mark areas you want to restore:")
    
    # Canvas setup
    stroke_width = st.sidebar.slider("Brush Size:", 1, 25, 5)
    h, w = image.shape[:2]
    
    # Resize for canvas if too large
    if w > 800:
        canvas_h, canvas_w = int(h * 800 / w), 800
    else:
        canvas_h, canvas_w = h, w
    
    # Create canvas
    canvas_result = st_canvas(
        fill_color='rgba(0, 0, 0, 0)',
        stroke_width=stroke_width,
        stroke_color='#FF0000',
        background_image=Image.open(uploaded_file).resize((canvas_w, canvas_h)),
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode='freedraw',
        key="inpainting_canvas",
    )
    
    if canvas_result.image_data is not None:
        # Process mask
        mask = canvas_result.image_data[:, :, 3]  # Alpha channel
        mask = np.uint8(mask > 0) * 255  # Binary mask
        mask = cv2.resize(mask, (w, h))
        
        if st.sidebar.checkbox('👁️ Show Mask'):
            st.subheader("Inpainting Mask")
            st.image(mask, caption="Red areas will be inpainted")
        
        # Inpainting method selection
        st.sidebar.markdown("---")
        inpaint_method = st.sidebar.selectbox(
            '🔧 Inpainting Algorithm:',
            ['None', 'Telea', 'Navier-Stokes', 'Compare Both'],
            help="Choose the inpainting algorithm to use"
        )
        
        if inpaint_method != 'None' and np.any(mask):
            if inpaint_method == 'Compare Both':
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Telea Method")
                    result_telea = processor.apply_inpainting(image, mask, 'telea')
                    result_telea_rgb = cv2.cvtColor(result_telea, cv2.COLOR_BGR2RGB)
                    st.image(result_telea_rgb)
                
                with col2:
                    st.subheader("Navier-Stokes Method")
                    result_ns = processor.apply_inpainting(image, mask, 'ns')
                    result_ns_rgb = cv2.cvtColor(result_ns, cv2.COLOR_BGR2RGB)
                    st.image(result_ns_rgb)
                
                # Download links for both
                st.sidebar.markdown("---")
                st.sidebar.markdown("📥 **Download Results:**")
                st.sidebar.markdown(
                    ui.create_download_link(Image.fromarray(result_telea_rgb), 
                                          'inpaint_telea.jpg', 'Telea Result'),
                    unsafe_allow_html=True
                )
                st.sidebar.markdown(
                    ui.create_download_link(Image.fromarray(result_ns_rgb), 
                                          'inpaint_ns.jpg', 'NS Result'),
                    unsafe_allow_html=True
                )
            
            else:
                method = 'telea' if inpaint_method == 'Telea' else 'ns'
                result = processor.apply_inpainting(image, mask, method)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                with col2:
                    st.subheader(f"Inpainted - {inpaint_method}")
                    st.image(result_rgb)
                
                # Download link
                st.sidebar.markdown("---")
                st.sidebar.markdown(
                    ui.create_download_link(Image.fromarray(result_rgb), 
                                          f'inpaint_{method}.jpg', 
                                          f'📥 Download {inpaint_method} Result'),
                    unsafe_allow_html=True
                )
        
        elif inpaint_method != 'None' and not np.any(mask):
            st.warning("⚠️ Please draw on the image to create a mask for inpainting.")


if __name__ == "__main__":
    main()