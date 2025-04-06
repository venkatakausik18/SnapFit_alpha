import torch
import numpy as np
from PIL import Image
import base64
import io
import os
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16

# Corrected path for Serverless Endpoints
MODEL_PATH = "/runpod-volume/checkpoints"

# Replace async def load_models_async() with:
def load_models():
    """Synchronous model loader"""
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=torch_dtype
    )
    
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype
    )
    
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_PATH,
        subfolder="transformer",
        torch_dtype=torch_dtype
    )
    
    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH,
        subfolder="vae"
    )

    pipe = FluxTryonPipeline.from_pretrained(
        MODEL_PATH,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    ).to(device=device, dtype=torch_dtype)

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    try:
        pipe.load_lora_weights(
            "loooooong/Any2anyTryon",
            weight_name="dev_lora_any2any_tryon.safetensors",
            adapter_name="tryon"
        )
    except Exception as e:
        print(f"LoRA weights not loaded: {e}")

    return pipe  # Return the pipeline object


@app.on_event("startup")
async def startup_event():
    """Async model loading during startup"""
    await load_models_async()

class TryOnRequest(BaseModel):
    user_image_base64: str
    garment_image_base64: str

def generate_image(model_image: np.ndarray, garment_image: np.ndarray, height=512, width=384, seed=0, guidance_scale=3.5, num_inference_steps=30):
    height, width = int(height), int(width)
    width = width - (width % 16)
    height = height - (height % 16)

    concat_image_list = [np.zeros((height, width, 3), dtype=np.uint8)]
    has_model_image = model_image is not None
    has_garment_image = garment_image is not None

    if has_model_image:
        if model_image.shape[-1] == 4:
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

    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)

    mask = np.zeros_like(np.array(image))
    mask[:, :width] = 255
    mask_image = Image.fromarray(mask)

    output = app.pipe(
        "",
        image=image,
        mask_image=mask_image,
        strength=1.0,
        height=height,
        width=image.width,
        target_width=width,
        tryon=has_model_image and has_garment_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="pil",
    ).images[0]

    return output

def process_images_standalone(user_image_base64: str, garment_image_base64: str):
    try:
        user_image_data = base64.b64decode(user_image_base64)
        garment_image_data = base64.b64decode(garment_image_base64)

        user_img = Image.open(io.BytesIO(user_image_data)).convert("RGBA")
        garment_img = Image.open(io.BytesIO(garment_image_data)).convert("RGBA")

        user_img_np = np.array(user_img)
        garment_img_np = np.array(garment_img)

        output_image = generate_image(model_image=user_img_np, garment_image=garment_img_np)

        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")

        return {"output_image": output_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ready" if hasattr(app, 'pipe') else "loading"}

@app.post("/try-on/", response_model=dict)
async def try_on(request: TryOnRequest):
    return process_images_standalone(request.user_image_base64, request.garment_image_base64)

@app.get("/")
async def root():
    return {"message": "Welcome to the Virtual Try-On API. Use POST /try-on/ with base64 image strings."}



__all__ = ['load_models', 'TryOnRequest', 'process_images_standalone']

