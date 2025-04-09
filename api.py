import torch
import numpy as np
from PIL import Image
import base64
import io
from fastapi import FastAPI, HTTPException, File, UploadFile  # <-- Add File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse  # <-- Add FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline_tryon import FluxTryonPipeline, resize_by_height
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxTransformer2DModel, AutoencoderKL
from fastapi.staticfiles import StaticFiles
from io import BytesIO
# Initialize FastAPI app
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16

def load_models(device=device, torch_dtype=torch_dtype):
    local_repo = "/workspace/checkpoints"
    text_encoder = CLIPTextModel.from_pretrained(local_repo, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(local_repo, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(local_repo, subfolder="transformer", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(local_repo, subfolder="vae")
    
    pipe = FluxTryonPipeline.from_pretrained(
        local_repo,
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

# Load models at startup
pipe = load_models()

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
        if garment_image.shape[-1] == 4:  # Convert RGBA to RGB if needed
            garment_image = Image.fromarray(garment_image, mode="RGBA").convert("RGB")
            garment_image = np.array(garment_image)
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)
    
    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)
    
    mask = np.zeros_like(np.array(image))
    mask[:, :width] = 255
    mask_image = Image.fromarray(mask)
    
    output = pipe(
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
# Serve the frontend at root

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

# Endpoint for virtual try-on
@app.post("/try-on/")
async def try_on(model_image: UploadFile = File(...), garment_image: UploadFile = File(...)):
    # Read and process model image
    model_image_data = await model_image.read()
    model_image = Image.open(BytesIO(model_image_data))
    model_image = np.array(model_image)
    
    # Read and process garment image
    garment_image_data = await garment_image.read()
    garment_image = Image.open(BytesIO(garment_image_data))
    garment_image = np.array(garment_image)
    
    # Generate the try-on image
    output_image = generate_image(model_image=model_image, garment_image=garment_image)
    
    # Save temporarily and return
    output_path = "generated_image.png"
    output_image.save(output_path)
    return FileResponse(output_path)
