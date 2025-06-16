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

def debug_canvas_setup():
    """Debug function to test canvas component."""
    st.subheader("🔧 Canvas Debug Mode")
    
    # Create a simple test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (50, 50), (350, 250), (255, 0, 0), 3)
    cv2.putText(test_image, "Test Image", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    pil_test = Image.fromarray(test_image)
    
    try:
        canvas_result = st_canvas(
            fill_color='rgba(0, 0, 0, 0)',
            stroke_width=5,
            stroke_color='#00FF00',
            background_image=pil_test,
            update_streamlit=True,
            height=300,
            width=400,
            drawing_mode='freedraw',
            key="debug_canvas",
        )
        
        if canvas_result.image_data is not None:
            st.success("✅ Canvas is working!")
            st.image(canvas_result.image_data, caption="Canvas output")
        else:
            st.warning("Canvas created but no image data")
            
    except Exception as e:
        st.error(f"Canvas failed: {e}")

def handle_inpainting(image: np.ndarray, uploaded_file, 
                     processor: ImageProcessor, ui: UIComponents):
    """Enhanced inpainting with multiple fallback methods for background image."""
    st.subheader("🎨 Interactive Inpainting")
    st.markdown("Draw on the image to mark areas you want to restore:")
    
    # Multiple approaches to ensure background image works
    h, w = image.shape[:2]
    
    # Method 1: Try to use the uploaded file directly
    background_image = None
    canvas_key = f"canvas_{uploaded_file.name}_{uploaded_file.size}"
    
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read file as bytes and create PIL image
        file_bytes = uploaded_file.read()
        background_image = Image.open(io.BytesIO(file_bytes))
        
        # Resize for canvas display
        if w > 800:
            canvas_h, canvas_w = int(h * 800 / w), 800
        else:
            canvas_h, canvas_w = h, w
            
        background_image = background_image.resize((canvas_w, canvas_h))
        
    except Exception as e:
        st.warning(f"Could not load background from uploaded file: {e}")
        
        # Method 2: Convert OpenCV image to PIL as fallback
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if w > 800:
                canvas_h, canvas_w = int(h * 800 / w), 800
                image_rgb = cv2.resize(image_rgb, (canvas_w, canvas_h))
            else:
                canvas_h, canvas_w = h, w
            background_image = Image.fromarray(image_rgb)
            
        except Exception as e2:
            st.error(f"Failed to create background image: {e2}")
            return
    
    # Show background preview for debugging
    if st.sidebar.checkbox("🖼️ Show Background Preview"):
        st.sidebar.image(background_image, caption="Canvas Background", width=200)
        st.sidebar.write(f"Background size: {background_image.size}")
        st.sidebar.write(f"Canvas size: {canvas_w} x {canvas_h}")
    
    # Canvas controls
    stroke_width = st.sidebar.slider("Brush Size:", 1, 25, 5)
    
    # Create canvas with multiple fallback options
    canvas_result = None
    
    # Try different canvas configurations
    for attempt in range(3):
        try:
            if attempt == 0:
                # Standard approach
                canvas_result = st_canvas(
                    fill_color='rgba(0, 0, 0, 0)',
                    stroke_width=stroke_width,
                    stroke_color='#FF0000',
                    background_color='',
                    background_image=background_image,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode='freedraw',
                    key=f"{canvas_key}_attempt_{attempt}",
                    display_toolbar=True,
                )
            elif attempt == 1:
                # With white background fallback
                canvas_result = st_canvas(
                    fill_color='rgba(0, 0, 0, 0)',
                    stroke_width=stroke_width,
                    stroke_color='#FF0000',
                    background_color='#FFFFFF',
                    background_image=background_image,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode='freedraw',
                    key=f"{canvas_key}_attempt_{attempt}",
                    display_toolbar=True,
                )
            else:
                # Simplified version without background
                st.warning("Background image failed to load. Using simplified canvas.")
                canvas_result = st_canvas(
                    fill_color='rgba(255, 255, 255, 0.8)',
                    stroke_width=stroke_width,
                    stroke_color='#FF0000',
                    background_color='#F0F0F0',
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode='freedraw',
                    key=f"{canvas_key}_simple",
                    display_toolbar=True,
                )
                # Show original image separately
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Original Image (draw mask on canvas above)", 
                        width=canvas_w)
            
            # If we get here, canvas was created successfully
            break
            
        except Exception as canvas_error:
            st.warning(f"Canvas attempt {attempt + 1} failed: {canvas_error}")
            if attempt == 2:  # Last attempt
                st.error("Canvas component failed to load. Please refresh the page.")
                return
    
    # Process canvas result
    if canvas_result and canvas_result.image_data is not None:
        # Extract mask from canvas
        mask_data = canvas_result.image_data
        
        # Check if we have valid mask data
        if mask_data.shape[-1] >= 4:  # Has alpha channel
            mask = mask_data[:, :, 3]  # Alpha channel
        else:
            # Fallback: use any non-white pixels as mask
            mask = np.any(mask_data[:, :, :3] != [255, 255, 255], axis=2).astype(np.uint8) * 255
        
        # Convert to binary mask
        mask = np.uint8(mask > 0) * 255
        
        # Resize mask back to original image size
        mask = cv2.resize(mask, (w, h))
        
        # Show mask if requested
        if st.sidebar.checkbox('👁️ Show Mask'):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader("Inpainting Mask")
                st.image(mask, caption="White areas will be inpainted")
        
        # Inpainting controls
        st.sidebar.markdown("---")
        inpaint_method = st.sidebar.selectbox(
            '🔧 Inpainting Algorithm:',
            ['None', 'Telea', 'Navier-Stokes', 'Compare Both'],
            help="Choose the inpainting algorithm to use"
        )
        
        # Apply inpainting
        if inpaint_method != 'None' and np.any(mask):
            if st.sidebar.button("🚀 Apply Inpainting", type="primary"):
                
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
                    
                    with col2:
                        st.subheader("Navier-Stokes Method")
                        with st.spinner("Processing with Navier-Stokes..."):
                            result_ns = safe_process_image(
                                processor.apply_inpainting, image, mask, 'ns'
                            )
                            if result_ns is not None:
                                result_ns_rgb = cv2.cvtColor(result_ns, cv2.COLOR_BGR2RGB)
                                st.image(result_ns_rgb)
                    
                    # Download links
                    if 'result_telea_rgb' in locals() and 'result_ns_rgb' in locals():
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
                            
                            # Download link
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
            st.warning("⚠️ Please draw on the canvas to create a mask for inpainting.")
    
    else:
        st.info("Canvas is loading... If issues persist, try refreshing the page.")
        
        # Alternative: Provide non-canvas inpainting options
        with st.expander("🔧 Alternative: Upload Mask Method"):
            st.markdown("If canvas isn't working, you can upload a mask image instead:")
            mask_file = st.file_uploader("Upload black/white mask (white = inpaint)", 
                                       type=["png", "jpg"], key="mask_upload")
            if mask_file:
                mask_bytes = np.asarray(bytearray(mask_file.read()), dtype=np.uint8)
                mask_img = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask_img, (w, h))
                
                st.image(mask, caption="Uploaded mask")
                
                method = st.selectbox("Algorithm:", ["Telea", "Navier-Stokes"], key="alt_method")
                if st.button("Apply Alternative Inpainting"):
                    result = safe_process_image(
                        processor.apply_inpainting, image, mask, method.lower()
                    )
                    if result is not None:
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, caption=f"Result - {method}")

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

        if st.sidebar.button("🔧 Test Canvas"):
            debug_canvas_setup()
        
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