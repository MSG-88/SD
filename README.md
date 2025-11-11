# Diffusion Image Endpoint

This project exposes a FastAPI + Streamlit application powered by [🤗 Diffusers](https://huggingface.co/docs/diffusers/index).  
You can switch between multiple Hugging Face checkpoints (e.g., `UnfilteredAI/NSFW-gen-v2`, `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell`) and optionally layer custom LoRA adapters (such as `lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1`) to change the style of generated images or short GIFs.

## Project Structure

```
safetensor-image-endpoint
├── src
│   ├── app.py                # Entry point of the application
│   ├── api
│   │   ├── endpoints.py      # API endpoint definitions
│   │   └── deps.py           # Dependency functions
│   ├── core
│   │   ├── config.py         # Application configuration management
│   ├── services
│   │   ├── diffusion.py      # Model registry + Diffusers pipeline management
│   │   └── inference.py      # Prompt → media orchestration
│   └── schemas
│       └── request.py        # Request schema definitions and enums
├── tests
│   └── test_endpoints.py     # Unit tests for API endpoints
├── Dockerfile                 # Docker image build instructions
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project dependencies and configurations
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd safetensor-image-endpoint
   ```

2. **Install dependencies (includes diffusers + accelerate):**
   ```
   pip install -r requirements.txt
   ```

3. **Run the FastAPI backend (optional when using Streamlit locally):**
   ```
   uvicorn src.app:app --reload
   ```

4. **Launch the Streamlit UI:**
   ```
   streamlit run src/app.py
   ```
   The sidebar exposes the list of registered Hugging Face / LoRA presets so you can swap styles on the fly.

5. **Access the API directly (if desired):**
   The API lives at `http://localhost:8000` with two endpoints under `/api/v1`:
   - `GET /api/v1/models` – metadata about every registered checkpoint (including LoRA info).
   - `POST /api/v1/generate` – create an image or GIF from a prompt (set `model_name` to select the preset).

6. **Hugging Face authentication:**
   The application reads the Hugging Face token from environment variables or a local `.env` file. Set `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_TOKEN` in your environment or copy `.env.example` to `.env` and add your token there.

   IMPORTANT: Never commit real API tokens to source control. This repository includes `.env.example` as a template and `.gitignore` excludes `.env`.

### Model Configuration & LoRA Support

`src/services/diffusion.py` defines several presets out of the box:

| Key | Base Model | Pipeline | Notes |
| --- | --- | --- | --- |
| `nsfw-gen-v2` | `UnfilteredAI/NSFW-gen-v2` | `DiffusionPipeline` | General-purpose NSFW generator |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | `AutoPipelineForText2Image` | Loads the `lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1` LoRA with weight `flux_lustly-ai_v1.safetensors` |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | `AutoPipelineForText2Image` | Fast flux preset without LoRA |

Each entry captures device, dtype, inference steps, resolution, and optional LoRA metadata.  
Extend `DEFAULT_MODEL_CONFIGS` with your own Hugging Face repos or adapters, then select them via:

- `model_name=<preset-key>` in API/Streamlit requests
- `DEFAULT_MODEL_NAME=<preset-key>` in `.env` to change the default selection

## Usage Examples

### Prompt → Image

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -F "prompt=A neon cyberpunk skyline" \
     -F "model_name=nsfw-gen-v2" \
     -F "input_type=prompt" \
     -F "output_type=image"
```

### Image → Video

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
     -F "prompt=Transform this sketch into a looping animation" \
     -F "model_name=flux-dev" \
     -F "input_type=prompt" \
     -F "output_type=video" \
     -F "frames=8" \
```

Responses contain Base64 encoded media plus metadata describing which preset (including LoRA adapter), prompt, seed, and frame count were used.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
