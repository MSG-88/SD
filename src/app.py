from __future__ import annotations
import sys
from pathlib import Path
from fastapi import FastAPI

# Ensure the project root is importable when this file is executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.deps import get_inference_service
from src.api.endpoints import router as api_router
from src.core.config import settings
from src.core.hf import ensure_hf_login
from src.schemas.request import GenerationParams, InputType, OutputType

app = FastAPI(title=settings.project_name)
app.include_router(api_router)


@app.on_event("startup")
def _startup_event() -> None:
    """Run once when the FastAPI app starts under an ASGI server.

    Ensures the Hugging Face token (if present) is registered with huggingface_hub.
    """
    ensure_hf_login()


def run_streamlit_ui() -> None:
    """Launch the Streamlit experience for prompt-based generations."""
    import base64
    import io
    from typing import List, Optional

    import streamlit as st
    from PIL import Image, UnidentifiedImageError

    service = get_inference_service()

    def _gather_models() -> List[dict]:
        return list(service.available_models())

    def _prepare_image(upload) -> Optional[Image.Image]:
        if upload is None:
            return None
        try:
            return Image.open(upload).convert("RGB")
        except UnidentifiedImageError:
            st.error("Uploaded file is not a valid image.")
        return None

    def _render_media(payload) -> None:
        media_bytes = base64.b64decode(payload["data"])
        label = f"Generated {payload['result_type'].capitalize()}"

        if payload["result_type"] == OutputType.image.value:
            image = Image.open(io.BytesIO(media_bytes))
            st.image(image, caption=label, use_column_width=True)
        else:
            st.image(media_bytes, caption=label, use_column_width=True)

        st.write("Metadata")
        st.json(payload["metadata"])

    st.set_page_config(page_title="Diffusion Studio", layout="wide")
    st.title("Diffusion Studio")
    st.caption("Generate images or gifs from prompts using Hugging Face diffusers.")

    with st.sidebar:
        st.header("Generation Settings")

        prompt = st.text_area("Prompt", value="A colorful abstract texture", height=120)
        uploaded_image = st.file_uploader(
            "Reference image (optional)", type=["png", "jpg", "jpeg", "webp"]
        )

        output_type_label = st.selectbox(
            "Output type",
            options=[OutputType.image, OutputType.video],
            format_func=lambda opt: opt.value,
        )
        frames = 8
        if output_type_label == OutputType.video:
            frames = st.slider("Frames (for GIFs)", min_value=2, max_value=24, value=8)

        seed = st.number_input("Seed (optional)", min_value=0, step=1, value=0)
        seed_value = seed if st.checkbox("Lock seed", value=False) else None

        model_options = _gather_models()
        if not model_options:
            st.warning("No models available.", icon="⚠️")
            selected_model_key = None
        else:
            label_map = {
                model["key"]: f"{model['name']} ({model['model_id']})"
                for model in model_options
            }
            keys = list(label_map.keys())
            default_key = (
                settings.default_model_name
                if settings.default_model_name in label_map
                else keys[0]
            )
            selected_model_key = st.selectbox(
                "Model",
                options=keys,
                format_func=lambda key: label_map[key],
                index=keys.index(default_key),
            )

    source_image = _prepare_image(uploaded_image)
    use_image_input = source_image is not None
    params = GenerationParams(
        prompt=prompt.strip(),
        input_type=InputType.image if use_image_input else InputType.prompt,
        output_type=output_type_label,
        model_name=selected_model_key,
        frames=frames,
        seed=seed_value,
    )

    if st.button("Generate", type="primary"):
        if not params.prompt:
            st.error("Prompt cannot be empty.")
            return

        try:
            payload = service.generate(params, source_image)
        except ValueError as exc:
            st.error(str(exc))
            return

        _render_media(payload)


def main() -> None:
    # Ensure Hugging Face token (if present) is logged for downstream libraries.
    ensure_hf_login()
    run_streamlit_ui()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Safetensor Image Processing API"}


if __name__ == "__main__":
    main()
