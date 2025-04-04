import torch
import numpy as np
from PIL import Image
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline_tryon import FluxTryonPipeline, resize_by_height
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxTransformer2DModel, AutoencoderKL

# Initialize FastAPI app
app = FastAPI(title="Virtual Try-On API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16

# Load models from the local cache directory (network volume)
def load_models(device=device, torch_dtype=torch_dtype):
    local_path = "/workspace/checkpoints"
    text_encoder = CLIPTextModel.from_pretrained(local_path, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(local_path, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(local_path, subfolder="transformer", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(local_path, subfolder="vae")

    pipe = FluxTryonPipeline.from_pretrained(
        local_path,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    ).to(device=device, dtype=torch_dtype)

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    pipe.load_lora_weights(
        "loooooong/Any2anyTryon",
        weight_name="dev_lora_any2any_tryon.safetensors",
        adapter_name="tryon",
    )
    return pipe

# Global pipeline variable (loaded at startup)
pipe = load_models()

# Define input model using Pydantic for request validation
class TryOnRequest(BaseModel):
    user_image_base64: str
    garment_image_base64: str

# Define the generate_image function (unchanged)
def generate_image(model_image: np.ndarray, garment_image: np.ndarray, height=512, width=384, seed=0, guidance_scale=3.5, num_inference_steps=30):
    height, width = int(height), int(width)
    width = width - (width % 16)
    height = height - (height % 16)

    concat_image_list = [np.zeros((height, width, 3), dtype=np.uint8)]
    has_model_image = model_image is not None
    has_garment_image = garment_image is not None

    if has_model_image:
        if model_image.shape[-1] == 4:  # Convert RGBA to RGB if needed
            model_image = Image.fromarray(model_image, mode="RGBA").convert("RGB")
            model_image = np.array(model_image)
        model_image = resize_by_height(model_image, height)
        concat_image_list.append(model_image)

    if has_garment_image:
        if garment_image.shape[-1] == 4:
            garment_image = Image.fromarray(garment_image, mode="RGBA").convert("RGB")
            garment_image = np.array(garment_image)
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)

    concat_image = np.concatenate(concat_image_list, axis=1)
    concat_image = Image.fromarray(concat_image)
    result = pipe(
        image=concat_image,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    ).images[0]

    return result

# Define the POST route
@app.post("/generate")
async def generate(request: TryOnRequest):
    try:
        user_image = Image.open(io.BytesIO(base64.b64decode(request.user_image_base64.split(",")[1])))
        user_image_np = np.array(user_image)

        garment_image = Image.open(io.BytesIO(base64.b64decode(request.garment_image_base64.split(",")[1])))
        garment_image_np = np.array(garment_image)

        result_image = generate_image(user_image_np, garment_image_np)
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {"output_base64": f"data:image/png;base64,{encoded_image}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
