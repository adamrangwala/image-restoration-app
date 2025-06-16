import streamlit as st
import cv2
import numpy as np
import io
import base64
import gc
from PIL import Image
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Image Restoration App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
            processed = safe_process_image(
                processor.apply_median_blur, 
                image, 
                controls['kernel_size']
            )
        else:  # Bilateral Blur
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
            
            # Download link
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


def handle_inpainting(image: np.ndarray, uploaded_file, 
                           processor: ImageProcessor, ui: UIComponents):
    """Simplified inpainting without interactive canvas."""
    st.subheader("🎨 Image Inpainting")
    
    # Offer preset mask options
    mask_option = st.selectbox(
        "Choose mask creation method:",
        ["Upload Mask Image", "Create Simple Shapes"]
    )
    
    if mask_option == "Upload Mask Image":
        st.markdown("Upload a black and white mask image (white areas will be inpainted):")
        mask_file = st.file_uploader("Upload mask", type=["png", "jpg"], key="mask")
        
        if mask_file:
            # Load mask
            mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
            mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
            h, w = image.shape[:2]
            mask = cv2.resize(mask_img, (w, h))
            
            # Show mask
            if st.checkbox('👁️ Show Mask'):
                st.image(mask, caption="Inpainting mask")
            
            # Apply inpainting
            method = st.selectbox("Choose method:", ["Telea", "Navier-Stokes"])
            if st.button("Apply Inpainting"):
                result = safe_process_image(
                    processor.apply_inpainting, 
                    image, mask, method.lower()
                )
                if result is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original")
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    with col2:
                        st.subheader(f"Result - {method}")
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb)
                        
                        # Download link
                        result_pil = Image.fromarray(result_rgb)
                        st.sidebar.markdown(
                            ui.create_download_link(
                                result_pil, 
                                f'inpaint_{method.lower()}.jpg',
                                f'📥 Download Result'
                            ),
                            unsafe_allow_html=True
                        )
    
    else:  # Create Simple Shapes
        st.markdown("Create a simple rectangular mask:")
        col1, col2 = st.columns(2)
        
        h, w = image.shape[:2]
        with col1:
            x1 = st.slider("Left X", 0, w-1, w//4)
            y1 = st.slider("Top Y", 0, h-1, h//4)
        with col2:
            x2 = st.slider("Right X", 0, w-1, 3*w//4)
            y2 = st.slider("Bottom Y", 0, h-1, 3*h//4)
        
        # Create rectangular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        if st.checkbox('👁️ Show Mask'):
            st.image(mask, caption="Rectangular mask")
        
        # Apply inpainting
        method = st.selectbox("Choose method:", ["Telea", "Navier-Stokes"], key="rect_method")
        if st.button("Apply Inpainting", key="rect_inpaint"):
            result = safe_process_image(
                processor.apply_inpainting, 
                image, mask, method.lower()
            )
            if result is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                with col2:
                    st.subheader(f"Result - {method}")
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb)
                    
                    # Download link
                    result_pil = Image.fromarray(result_rgb)
                    st.sidebar.markdown(
                        ui.create_download_link(
                            result_pil, 
                            f'inpaint_{method.lower()}.jpg',
                            f'📥 Download Result'
                        ),
                        unsafe_allow_html=True
                    )
def main():
    """Main application function."""
    # Header
    st.title("🖼️ Interactive Image Restoration")
    st.markdown("""
    Transform your images using advanced computer vision techniques. 
    Upload an image and choose from various restoration methods.
    """)
    
    # Add performance info
    add_performance_info()
    
    # Initialize components
    processor = ImageProcessor()
    ui = UIComponents()
    
    # Setup sidebar
    sidebar_data = ui.setup_sidebar()
    uploaded_file = sidebar_data["uploaded_file"]
    
    # Show deployment info
    show_deployment_info()
    
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
        # Validate file size first
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
            st.error("File too large! Please upload an image smaller than 50MB.")
            return
        
        # Load image with caching
        original_image = load_and_process_image(uploaded_file)
        
        if original_image is None:
            st.error("Failed to load image. Please try a different file.")
            return
        
        # Display image warnings
        display_image_warning(original_image)
        
        # Offer to resize large images
        if not validate_image_size(original_image):
            if st.button("🔄 Resize image for better performance"):
                original_image = resize_large_image(original_image)
                st.success("Image resized successfully!")
        
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


if __name__ == "__main__":
    main()