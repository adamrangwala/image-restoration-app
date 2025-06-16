"""
Interactive Image Restoration Application

A comprehensive Streamlit application for image restoration using various
computer vision techniques including median blur, bilateral filtering,
and advanced inpainting algorithms.

Author: [Your Name]
Date: [Current Date]
"""

import streamlit as st
import cv2
import numpy as np
import io
import base64
import gc
import requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas
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


# Sample image loading functions
@st.cache_data
def load_sample_from_github():
    """Load sample image from GitHub repository assets folder."""
    # Replace with your actual GitHub repository URL
    GITHUB_REPO_URL = "https://github.com/adamrangwala/image-restoration-app/main"
    SAMPLE_IMAGE_PATH = "/assets/old_image.jpg"
    
    sample_url = GITHUB_REPO_URL + SAMPLE_IMAGE_PATH
    
    try:
        # Download image from GitHub
        response = requests.get(sample_url, timeout=10)
        
        if response.status_code == 200:
            # Convert bytes to PIL Image
            image_bytes = io.BytesIO(response.content)
            pil_image = Image.open(image_bytes)
            
            # Convert PIL to OpenCV format
            image_array = np.array(pil_image)
            if len(image_array.shape) == 3:
                # Convert RGB to BGR for OpenCV
                opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale image
                opencv_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            return opencv_image, image_bytes.getvalue()
        else:
            st.warning(f"Could not load sample image. Status code: {response.status_code}")
            return None, None
            
    except requests.exceptions.RequestException as e:
        st.warning(f"Network error loading sample image: {e}")
        return None, None
    except Exception as e:
        st.warning(f"Error processing sample image: {e}")
        return None, None


class GitHubSampleFile:
    """File-like object for GitHub sample images."""
    def __init__(self, image_bytes, filename="sample_image.jpg"):
        self.name = filename
        self.size = len(image_bytes)
        self._bytes = image_bytes
        self._position = 0
    
    def read(self):
        return self._bytes
    
    def seek(self, pos):
        self._position = pos


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


def setup_sidebar_with_sample():
    """Configure sidebar with upload and GitHub sample option."""
    st.sidebar.title('🖼️ Image Restoration Toolkit')
    st.sidebar.markdown("---")
    
    # Sample image option
    if st.sidebar.button("🎨 Try Sample Image", help="Load a sample image from our collection"):
        st.session_state.use_sample = True
    
    # Upload option
    uploaded_file = st.sidebar.file_uploader(
        "📁 Or Upload Your Image:",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    # If file is uploaded, clear sample
    if uploaded_file is not None:
        st.session_state.use_sample = False
    
    # Get sample image if requested
    sample_image = None
    sample_file = None
    
    if st.session_state.get('use_sample', False):
        with st.spinner("Loading sample image..."):
            sample_image, sample_bytes = load_sample_from_github()
            
            if sample_image is not None:
                sample_file = GitHubSampleFile(sample_bytes)
                
                # Show sample preview
                if st.sidebar.checkbox("👁️ Preview Sample"):
                    sample_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
                    st.sidebar.image(sample_rgb, caption="Sample Image", width=200)
            else:
                st.sidebar.error("❌ Could not load sample image")
                st.session_state.use_sample = False
    
    return {
        "uploaded_file": uploaded_file, 
        "sample_image": sample_image,
        "sample_file": sample_file
    }


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
    """Inpainting with persistent background image handling."""
    st.subheader("🎨 Interactive Inpainting")
    st.markdown("Draw on the image to mark areas you want to restore:")
    
    # Store background image in session state to persist across widget changes
    if 'background_image' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
        # Reset file pointer and create background image
        uploaded_file.seek(0)
        pil_image = Image.open(io.BytesIO(uploaded_file.read()))
        
        # Store in session state
        st.session_state.background_image = pil_image
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.original_image = image.copy()  # Store original image too
    
    # Get canvas setup controls
    stroke_width = st.sidebar.slider("Brush Size:", 1, 25, 5)
    
    # Use stored background image
    background_image = st.session_state.background_image
    h, w = image.shape[:2]
    
    # Resize for canvas if too large
    if w > 800:
        canvas_h, canvas_w = int(h * 800 / w), 800
    else:
        canvas_h, canvas_w = h, w
    
    # Resize background image to match canvas
    resized_background = background_image.resize((canvas_w, canvas_h))
    
    # Create canvas with unique key that includes the image name to force refresh on new uploads
    canvas_key = f"inpainting_canvas_{st.session_state.last_uploaded_file}"
    
    canvas_result = st_canvas(
        fill_color='rgba(0, 0, 0, 0)',
        stroke_width=stroke_width,
        stroke_color='#FF0000',
        background_color='',
        background_image=resized_background,  # Use the persistent background
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode='freedraw',
        key=canvas_key,  # Dynamic key prevents conflicts
        display_toolbar=True,
    )
    
    # Debug: Show background preview
    if st.sidebar.checkbox("🖼️ Show Background Preview"):
        st.sidebar.image(resized_background, caption="Canvas Background", width=200)
    
    if canvas_result.image_data is not None:
        # Process mask
        mask = canvas_result.image_data[:, :, 3]  # Alpha channel
        mask = np.uint8(mask > 0) * 255  # Binary mask
        mask = cv2.resize(mask, (w, h))  # Resize back to original image size
        
        if st.sidebar.checkbox('👁️ Show Mask'):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader("Inpainting Mask")
                st.image(mask, caption="White areas will be inpainted")
        
        # Inpainting method selection
        st.sidebar.markdown("---")
        inpaint_method = st.sidebar.selectbox(
            '🔧 Inpainting Algorithm:',
            ['None', 'Telea', 'Navier-Stokes', 'Compare Both'],
            help="Choose the inpainting algorithm to use"
        )
        
        if inpaint_method != 'None' and np.any(mask):
            if st.button("🚀 Apply Inpainting", type="primary"):
                if inpaint_method == 'Compare Both':
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Telea Method")
                        with st.spinner("Processing with Telea..."):
                            result_telea = safe_process_image(
                                processor.apply_inpainting, image, mask, 'telea'
                            )
                            if result_telea is not None:
                                result_telea_rgb = cv2.cvtColor(result_telea, cv2.COLOR_BGR2RGB)
                                st.image(result_telea_rgb)
                                # Store result in session state
                                st.session_state.result_telea = result_telea_rgb
                    
                    with col2:
                        st.subheader("Navier-Stokes Method")
                        with st.spinner("Processing with Navier-Stokes..."):
                            result_ns = safe_process_image(
                                processor.apply_inpainting, image, mask, 'ns'
                            )
                            if result_ns is not None:
                                result_ns_rgb = cv2.cvtColor(result_ns, cv2.COLOR_BGR2RGB)
                                st.image(result_ns_rgb)
                                # Store result in session state
                                st.session_state.result_ns = result_ns_rgb
                    
                    # Download links for both (use session state)
                    if hasattr(st.session_state, 'result_telea') and hasattr(st.session_state, 'result_ns'):
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("📥 **Download Results:**")
                        st.sidebar.markdown(
                            ui.create_download_link(
                                Image.fromarray(st.session_state.result_telea), 
                                'inpaint_telea.jpg', 'Telea Result'
                            ),
                            unsafe_allow_html=True
                        )
                        st.sidebar.markdown(
                            ui.create_download_link(
                                Image.fromarray(st.session_state.result_ns), 
                                'inpaint_ns.jpg', 'NS Result'
                            ),
                            unsafe_allow_html=True
                        )
                
                else:
                    method = 'telea' if inpaint_method == 'Telea' else 'ns'
                    
                    with st.spinner(f"Processing with {inpaint_method}..."):
                        result = safe_process_image(
                            processor.apply_inpainting, image, mask, method
                        )
                        
                        if result is not None:
                            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Original")
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                            
                            with col2:
                                st.subheader(f"Inpainted - {inpaint_method}")
                                st.image(result_rgb)
                            
                            # Store result and show download link
                            st.session_state.current_result = result_rgb
                            st.sidebar.markdown("---")
                            st.sidebar.markdown(
                                ui.create_download_link(
                                    Image.fromarray(result_rgb), 
                                    f'inpaint_{method}.jpg',
                                    f'📥 Download {inpaint_method} Result'
                                ),
                                unsafe_allow_html=True
                            )
        
        elif inpaint_method != 'None' and not np.any(mask):
            st.warning("⚠️ Please draw on the image to create a mask for inpainting.")
    
    else:
        st.info("Draw on the image above to create an inpainting mask.")
    
    # Show previous results if they exist
    if hasattr(st.session_state, 'current_result'):
        with st.expander("📋 Previous Result"):
            st.image(st.session_state.current_result, caption="Last inpainting result")


def main():
    """Main application function with GitHub sample image support."""
    # Header
    st.title("🖼️ Interactive Image Restoration")
    st.markdown("""
    Transform your images using advanced computer vision techniques. 
    Upload your own image or try our sample to get started!
    """)
    
    # Add performance info
    add_performance_info()
    
    # Initialize components
    processor = ImageProcessor()
    ui = UIComponents()
    
    # Setup sidebar
    sidebar_data = setup_sidebar_with_sample()
    uploaded_file = sidebar_data["uploaded_file"]
    sample_image = sidebar_data["sample_image"]
    sample_file = sidebar_data["sample_file"]
    
    # Show deployment info
    show_deployment_info()
    
    # Determine which image to use
    original_image = None
    current_file = None
    
    if uploaded_file is not None:
        # Use uploaded file
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File too large! Please upload an image smaller than 50MB.")
            return
        
        original_image = load_and_process_image(uploaded_file)
        current_file = uploaded_file
        
    elif sample_image is not None and sample_file is not None:
        # Use sample image
        original_image = sample_image
        current_file = sample_file
    
    if original_image is None:
        # Display welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👆 Try our sample image or upload your own to get started!")
            st.markdown("""
            ### 🚀 Quick Start:
            1. Click **"Try Sample Image"** for an instant demo
            2. Or **upload your own image** to restore
            
            ### Available Restoration Methods:
            - **Median Blur**: Remove noise while preserving edges
            - **Bilateral Filter**: Edge-preserving smoothing  
            - **Image Inpainting**: Remove unwanted objects or restore damage
            """)
        return
    
    try:
        # Display image warnings
        display_image_warning(original_image)
        
        # Offer to resize large images
        if not validate_image_size(original_image):
            if st.button("🔄 Resize image for better performance"):
                original_image = resize_large_image(original_image)
                st.success("Image resized successfully!")
        
        # Display image info
        h, w, c = original_image.shape
        image_source = "Sample Image" if sample_image is not None else "Uploaded Image"
        st.sidebar.markdown(f"""
        **Image Info:**
        - Source: {image_source}
        - Dimensions: {w} × {h}
        - Channels: {c}
        """)
        
        if uploaded_file:
            st.sidebar.markdown(f"- Size: {uploaded_file.size / 1024:.1f} KB")
        elif sample_file:
            st.sidebar.markdown(f"- Size: {sample_file.size / 1024:.1f} KB")
        
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
            
            if sample_image is not None:
                st.markdown("""
                🎯 **Try the restoration methods above to see the magic in action!**
                
                Each method is designed for different types of image problems.
                """)
        
        elif option in ['Median Blur', 'Bilateral Blur']:
            handle_blur_filters(original_image, option, processor, ui)
        
        elif option == 'Image Inpainting':
            handle_inpainting(original_image, current_file, processor, ui)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"An error occurred while processing the image: {str(e)}")


if __name__ == "__main__":
    main()