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

# Page configuration
st.set_page_config(
    page_title="Image Restoration App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ImageProcessor:
    @staticmethod
    def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_bilateral_filter(image: np.ndarray, diameter: int, sigma_color: int, sigma_space: int) -> np.ndarray:
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    @staticmethod
    def apply_inpainting(image: np.ndarray, mask: np.ndarray, method: str = 'telea') -> np.ndarray:
        flag = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=flag)

class UIComponents:
    @staticmethod
    def create_download_link(img: Image.Image, filename: str, text: str) -> str:
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    
    @staticmethod
    def setup_sidebar() -> dict:
        st.sidebar.title('🖼️ Image Restoration Toolkit')
        st.sidebar.markdown("---")
        
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Image to Restore:",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        sample_image_button = st.sidebar.button("🖼️ Use Sample Image")
        
        return {"uploaded_file": uploaded_file, "sample_image_button": sample_image_button}
    
    @staticmethod
    def create_filter_controls(option: str) -> dict:
        controls = {}
        if option == 'Median Blur':
            controls['kernel_size'] = st.sidebar.slider("Kernel Size:", 3, 15, 5, 2)
        elif option == 'Bilateral Blur':
            controls['diameter'] = st.sidebar.slider("Diameter:", 1, 50, 20)
            controls['sigma_color'] = st.sidebar.slider("Sigma Color:", 0, 250, 200, 10)
            controls['sigma_space'] = st.sidebar.slider("Sigma Space:", 0, 250, 100, 10)
        return controls

@st.cache_data
def load_and_process_image(_uploaded_file):
    try:
        _uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(_uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

@st.cache_data
def validate_image_size(image: np.ndarray, max_dimension: int = 2048) -> bool:
    h, w = image.shape[:2]
    return max(h, w) <= max_dimension

def memory_cleanup():
    gc.collect()

def display_image_warning(image: np.ndarray):
    h, w = image.shape[:2]
    size_mb = (h * w * 3) / (1024 * 1024)
    if size_mb > 25:
        st.warning(f"⚠️ Large image detected ({size_mb:.1f}MB). Processing may be slow.")
    elif max(h, w) > 2048:
        st.warning("⚠️ Image is very large. Consider resizing for better performance.")

def resize_large_image(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > max_dimension:
        if h > w:
            new_h, new_w = max_dimension, int(w * max_dimension / h)
        else:
            new_h, new_w = int(h * max_dimension / w), max_dimension
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def safe_process_image(func, *args, **kwargs):
    try:
        with st.spinner("Processing image..."):
            result = func(*args, **kwargs)
            memory_cleanup()
            return result
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")
        return None

def add_performance_info():
    with st.expander("ℹ️ Performance Tips"):
        st.markdown("- Use images smaller than 2048x2048 pixels\n- JPEG format loads faster\n- Smaller brush sizes are faster for inpainting")

def show_deployment_info():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Live Demo**")
    st.sidebar.caption("Running on Streamlit Cloud")
    st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/adamrangwala/image-restoration-app)")

def handle_blur_filters(image: np.ndarray, option: str, processor: ImageProcessor, ui: UIComponents):
    controls = ui.create_filter_controls(option)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with col2:
        st.subheader(f"Result - {option}")
        if option == 'Median Blur':
            processed = safe_process_image(processor.apply_median_blur, image, controls['kernel_size'])
        else:
            processed = safe_process_image(processor.apply_bilateral_filter, image, controls['diameter'], controls['sigma_color'], controls['sigma_space'])
        if processed is not None:
            result_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(result_rgb)
            st.sidebar.markdown("---")
            st.sidebar.markdown(ui.create_download_link(Image.fromarray(result_rgb), f'{option.lower().replace(" ", "_")}.jpg', f'📥 Download {option} Result'), unsafe_allow_html=True)

def handle_inpainting(image: np.ndarray, file_info: dict, processor: ImageProcessor, ui: UIComponents):
    st.subheader("🎨 Interactive Inpainting")
    st.markdown("Draw on the image to mark areas for restoration:")
    
    h, w = image.shape[:2]
    
    mock_file = io.BytesIO(file_info['bytes'])
    mock_file.name = file_info['name']
    mock_file.size = file_info['size']
    
    try:
        background_image = Image.open(mock_file)
        if w > 800:
            canvas_h, canvas_w = int(h * 800 / w), 800
        else:
            canvas_h, canvas_w = h, w
        background_image = background_image.resize((canvas_w, canvas_h))
    except Exception as e:
        st.error(f"Failed to create background for canvas: {e}")
        return

    stroke_width = st.sidebar.slider("Brush Size:", 1, 25, 5)
    canvas_key = f"canvas_{file_info['name']}_{file_info['size']}"
    
    canvas_result = st_canvas(
        fill_color='rgba(0, 0, 0, 0)', stroke_width=stroke_width, stroke_color='#FF0000',
        background_image=background_image, update_streamlit=True, height=canvas_h,
        width=canvas_w, drawing_mode='freedraw', key=canvas_key, display_toolbar=True,
    )
    
    if canvas_result and canvas_result.image_data is not None:
        mask = cv2.resize(canvas_result.image_data[:, :, 3], (w, h))
        
        st.sidebar.markdown("---")
        inpaint_method = st.sidebar.selectbox('🔧 Inpainting Algorithm:', ['None', 'Telea', 'Navier-Stokes'])
        
        if inpaint_method != 'None' and np.any(mask):
            if st.sidebar.button("🚀 Apply Inpainting", type="primary"):
                result = safe_process_image(processor.apply_inpainting, image, mask, inpaint_method.lower())
                if result is not None:
                    st.subheader(f"Inpainted - {inpaint_method}")
                    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    st.sidebar.markdown("---")
                    st.sidebar.markdown(ui.create_download_link(Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), f'inpaint_{inpaint_method.lower()}.jpg', f'📥 Download Result'), unsafe_allow_html=True)
        elif inpaint_method != 'None':
            st.warning("⚠️ Please draw a mask on the image to apply inpainting.")

def main():
    st.title("🖼️ Interactive Image Restoration")
    st.markdown("Upload an image and choose from various restoration methods.")
    
    add_performance_info()
    processor = ImageProcessor()
    ui = UIComponents()

    # Initialize state keys
    if "image_data" not in st.session_state:
        st.session_state.image_data = None
        st.session_state.file_info = None
        st.session_state.last_upload_id = None
        st.session_state.source = None

    sidebar_data = ui.setup_sidebar()
    uploaded_file = sidebar_data["uploaded_file"]
    sample_button = sidebar_data["sample_image_button"]
    
    show_deployment_info()

    # --- State Update Logic ---
    # This section exclusively handles changing the image source.
    
    # A. User clicks the sample button
    if sample_button:
        try:
            with open("assets/old_image.jpg", "rb") as f:
                img_bytes = f.read()
            st.session_state.image_data = load_and_process_image(io.BytesIO(img_bytes))
            st.session_state.file_info = {'name': 'old_image.jpg', 'size': len(img_bytes), 'bytes': img_bytes}
            st.session_state.last_upload_id = None # Reset upload ID
            st.session_state.source = 'sample'
            st.rerun()
        except FileNotFoundError:
            st.error("Sample image 'assets/old_image.jpg' not found.")
            st.session_state.image_data = None

    # B. User uploads a new file
    elif uploaded_file and uploaded_file.id != st.session_state.get('last_upload_id'):
        st.session_state.image_data = load_and_process_image(uploaded_file)
        st.session_state.file_info = {'name': uploaded_file.name, 'size': uploaded_file.size, 'bytes': uploaded_file.getvalue()}
        st.session_state.last_upload_id = uploaded_file.id
        st.session_state.source = 'upload'
        st.rerun()

    # --- Display Logic ---
    # This section renders the UI based on the current session state.
    
    if st.session_state.image_data is None:
        st.info("👆 Please upload an image or use the sample to get started!")
        return

    original_image = st.session_state.image_data
    file_info = st.session_state.file_info
    
    display_image_warning(original_image)
    
    if not validate_image_size(original_image):
        if st.button("🔄 Resize image for better performance"):
            st.session_state.image_data = resize_large_image(original_image)
            st.rerun()
    
    h, w, _ = original_image.shape
    st.sidebar.markdown(f"**Image Info:**\n- Dimensions: {w} × {h}\n- Size: {file_info['size'] / 1024:.1f} KB")

    st.sidebar.markdown("---")
    option = st.sidebar.selectbox(
        '🎛️ Choose Restoration Method:',
        ('None', 'Median Blur', 'Bilateral Blur', 'Image Inpainting'),
        help="Select the image processing technique to apply"
    )
    
    if option == 'None':
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    elif option in ['Median Blur', 'Bilateral Blur']:
        handle_blur_filters(original_image, option, processor, ui)
    elif option == 'Image Inpainting':
        handle_inpainting(original_image, file_info, processor, ui)

if __name__ == "__main__":
    main()
