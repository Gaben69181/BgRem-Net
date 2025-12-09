"""frontend/app.py

Streamlit UI for background removal using U-Net segmentation.
"""

from __future__ import annotations

import io
import os
import base64
import sys
from typing import Optional, Tuple

from PIL import Image
import streamlit as st

# Make backend package importable when running `streamlit run frontend/app.py`
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default paths resolved relative to the project root
DEFAULT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_model.pth")
DEFAULT_TIMELAPSE_ROOT = os.path.join(PROJECT_ROOT, "outputs", "videos")

from backend.inference import (  # type: ignore  # noqa: E402
    InferenceConfig,
    load_model,
    predict_mask,
    compose_foreground,
)


@st.cache_resource(show_spinner=True)
def get_model(checkpoint_path: str, device: str, img_size: int, threshold: float):
    """Load and cache the U-Net model for inference."""
    cfg = InferenceConfig(
        checkpoint_path=checkpoint_path,
        device=device,
        img_size=img_size,
        threshold=threshold,
    )
    model = load_model(cfg)
    return model


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def main() -> None:
    st.set_page_config(page_title="Background Removal - U-Net Segmentation", layout="wide")

    st.title("Background Removal using U-Net Segmentation")
    st.markdown(
        "This demo performs **pixel-wise foreground segmentation** using a U-Net model "
        "trained on paired images and alpha mattes. Upload an image to remove the background, "
        "visualize the segmentation mask, and optionally apply a custom background color."
    )

    with st.sidebar:
        st.header("Configuration")

        checkpoint_path = st.text_input(
            "Model checkpoint path",
            value=DEFAULT_CHECKPOINT_PATH,
            help="Path to the trained U-Net checkpoint (.pth).",
        )

        img_size = st.number_input(
            "Internal resize (pixels)",
            min_value=256,
            max_value=1024,
            value=512,
            step=64,
            help="Input image will be resized to this resolution for the model.",
        )

        threshold = st.slider(
            "Mask threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Pixels with probability above this threshold are considered foreground.",
        )

        bg_mode = st.radio(
            "Background mode",
            options=["Transparent", "Solid color"],
            index=0,
            help="Select whether the background should be transparent or a solid color.",
        )

        bg_color_rgb: Optional[Tuple[int, int, int]] = None
        if bg_mode == "Solid color":
            bg_color = st.color_picker("Background color", value="#00FF00")
            # Convert hex to RGB tuple
            bg_color_rgb = tuple(int(bg_color[i : i + 2], 16) for i in (1, 3, 5))

        device = "cuda" if torch_available_and_cuda() else "cpu"
        st.caption(f"Inference device: **{device}**")

        st.markdown("---")
        st.subheader("Timelapse videos")
        st.caption(
            "After training with `--timelapse`, videos can be generated with "
            "`python backend/timelapse.py` and will appear under `outputs/videos`."
        )

        timelapse_root = DEFAULT_TIMELAPSE_ROOT
        timelapse_files = []
        if os.path.isdir(timelapse_root):
            for f in os.listdir(timelapse_root):
                if f.lower().endswith((".gif", ".mp4")):
                    timelapse_files.append(os.path.join(timelapse_root, f))
            timelapse_files.sort()

        selected_timelapse = None
        if timelapse_files:
            display_names = [os.path.basename(p) for p in timelapse_files]
            idx = st.selectbox(
                "Available timelapse videos",
                options=list(range(len(display_names))),
                format_func=lambda i: display_names[i],
            )
            selected_timelapse = timelapse_files[idx]
        else:
            st.caption("No timelapse videos found in `outputs/videos` yet.")

    uploaded_file = st.file_uploader(
        "Upload an input image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=False,
    )

    col1, col2, col3 = st.columns(3)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Failed to open image: {e}")
            return

        with st.spinner("Loading model..."):
            try:
                model = get_model(
                    checkpoint_path=checkpoint_path,
                    device=device,
                    img_size=img_size,
                    threshold=threshold,
                )
            except Exception as e:
                st.error(
                    "Failed to load model. Please check that the checkpoint path is correct "
                    "and that you have trained the model with `backend/train.py`."
                )
                st.exception(e)
                return

        with st.spinner("Running segmentation..."):
            try:
                bin_mask, soft_mask = predict_mask(
                    model,
                    image,
                    img_size=img_size,
                    threshold=threshold,
                )
                composed = compose_foreground(
                    image,
                    bin_mask,
                    background_color=bg_color_rgb,
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        # Ensure masks are in displayable size (already upscaled to original in predict_mask)
        with col1:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Segmentation mask")
            st.image(soft_mask, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Background removed")
            st.image(composed, use_container_width=True)

        st.markdown("---")
        st.subheader("Download results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.download_button(
                label="Download binary mask (PNG)",
                data=pil_to_bytes(bin_mask, format="PNG"),
                file_name="mask_binary.png",
                mime="image/png",
            )

        with col_b:
            st.download_button(
                label="Download soft mask (PNG)",
                data=pil_to_bytes(soft_mask, format="PNG"),
                file_name="mask_soft.png",
                mime="image/png",
            )

        with col_c:
            st.download_button(
                label="Download foreground (PNG)",
                data=pil_to_bytes(composed, format="PNG"),
                file_name="foreground.png",
                mime="image/png",
            )

    else:
        st.info("Upload an image using the file uploader above to start background removal.")

    st.markdown("---")
    st.subheader("Segmentation timelapse")

    if selected_timelapse is not None:
        st.write(f"Showing timelapse: `{os.path.basename(selected_timelapse)}`")
        if selected_timelapse.lower().endswith(".gif"):
            with open(selected_timelapse, "rb") as file_:
                contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="timelapse gif" style="max-width: 100%;">',
                unsafe_allow_html=True,
            )
        else:
            st.video(selected_timelapse)
    else:
        st.caption("Once you have generated timelapse videos, they will be listed here.")


def torch_available_and_cuda() -> bool:
    """Helper so we do not import torch at module import time for Streamlit."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()