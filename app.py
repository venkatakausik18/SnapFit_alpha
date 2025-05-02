import torch
import numpy as np
from PIL import Image
import base64
import io
import os
import gc
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.pipeline_tryon import FluxTryonPipeline, resize_by_height
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxTransformer2DModel, AutoencoderKL
from optimum.quanto import freeze, qfloat8, quantize

# Initialize FastAPI app
app = FastAPI(title="Virtual Try-On API")

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Set it to your GPU's compute capability
os.environ['CUDA_HOME'] = '/usr/local/cuda'  # Or try another common path
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16  # Changed from bfloat16 to float16 for better compatibility

# Global pipeline variable - will be lazily loaded
pipe = None

# Static watermark image path
WATERMARK_IMAGE_PATH = "/runpod-volume/mark.png"  # Path to the uploaded watermark image

# Load models (runs once at startup)
def load_models(device=device, torch_dtype=torch_dtype):
    global pipe
    print("Starting model loading process...")
    start_time = import_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    # Clear CUDA cache before loading models
    torch.cuda.empty_cache()
    
    # Check if we have a serialized model
    serialized_path = "/runpod-volume/serialized_models"  # Changed path
    os.makedirs(serialized_path, exist_ok=True)
    serialized_model_path = f"{serialized_path}/compiled_pipe.pt"
    
    if os.path.exists(serialized_model_path):
        print(f"Loading pre-compiled model from {serialized_model_path}")
        try:
            pipe = torch.load(serialized_model_path)
            print("Successfully loaded pre-compiled model")
            end_time.record()
            torch.cuda.synchronize()
            print(f"Model loading completed in {start_time.elapsed_time(end_time)/1000:.2f} seconds")
            return pipe
        except Exception as e:
            print(f"Error loading serialized model: {e}. Will load from original checkpoints.")
    
    # Load from original checkpoints if serialized model isn't available
    print("Loading models from original checkpoints")
    bfl_repo = "/runpod-volume/checkpoints"  # Changed path to match your environment
    
    # Load models with optimization flags
    text_encoder = CLIPTextModel.from_pretrained(
        bfl_repo, 
        subfolder="text_encoder", 
        torch_dtype=torch_dtype,
        local_files_only=True  # Prevent network lookups
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        bfl_repo, 
        subfolder="text_encoder_2", 
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        bfl_repo, 
        subfolder="transformer", 
        torch_dtype=torch_dtype,
        local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        bfl_repo, 
        subfolder="vae",
        local_files_only=True
    )
    
    pipe = FluxTryonPipeline.from_pretrained(
        bfl_repo,
        transformer=transformer,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        torch_dtype=torch_dtype,
    ).to(device=device, dtype=torch_dtype)
    
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    # Apply quantization (specifically for text_encoder_2 in this case)
    quantize(pipe.text_encoder_2, weights=qfloat8)
    freeze(pipe.text_encoder_2)  # Freeze model parameters for inference
    
    # Apply LoRA weights
    pipe.load_lora_weights(
        "loooooong/Any2anyTryon",
        weight_name="dev_lora_any2any_tryon.safetensors",
        adapter_name="tryon",
    )
    
    # Freeze other layers if necessary
    freeze(pipe.text_encoder)
    freeze(pipe.transformer)
    freeze(pipe.vae)
    
    # Save the compiled model for future use
    try:
        print(f"Saving compiled model to {serialized_model_path}")
        torch.save(pipe, serialized_model_path)
        print("Model successfully saved")
    except Exception as e:
        print(f"Failed to save compiled model: {e}")
    
    end_time.record()
    torch.cuda.synchronize()
    print(f"Model loading completed in {start_time.elapsed_time(end_time)/1000:.2f} seconds")
    return pipe

# Define input model using Pydantic for request validation
class TryOnRequest(BaseModel):
    user_image_base64: str
    garment_image_base64: str

# Define the generate_image function (with static watermark integration)
def generate_image_with_watermark(model_image: np.ndarray, garment_image: np.ndarray, height=512, width=384, seed=0, guidance_scale=3.5, num_inference_steps=30):
    global pipe
    # Lazy load models if not already loaded
    if pipe is None:
        pipe = load_models()
    
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
    
    # Load static watermark image
    watermark = Image.open(WATERMARK_IMAGE_PATH).convert("RGBA")
    watermark = watermark.resize((int(image.width * 0.2), int(image.height * 0.2)))  # Resize watermark to 20% of the image size
    watermark = watermark.convert("RGBA")
    
    # Position watermark at the bottom-right corner
    image.paste(watermark, (image.width - watermark.width - 10, image.height - watermark.height - 10), watermark)
    
    mask = np.zeros_like(np.array(image))
    mask[:, :width] = 255
    mask_image = Image.fromarray(mask)
    
    with torch.inference_mode():
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

# Define the processing function (adapted for API)
def process_images_standalone(user_image_base64: str, garment_image_base64: str):
    try:
        # Decode Base64 strings to images
        user_image_data = base64.b64decode(user_image_base64)
        garment_image_data = base64.b64decode(garment_image_base64)
        
        user_img = Image.open(io.BytesIO(user_image_data)).convert("RGBA")
        garment_img = Image.open(io.BytesIO(garment_image_data)).convert("RGBA")
        
        # Convert to NumPy arrays
        user_img_np = np.array(user_img)
        garment_img_np = np.array(garment_img)
        
        # Generate the try-on image with watermark
        output_image = generate_image_with_watermark(model_image=user_img_np, garment_image=garment_img_np)
        
        # Convert output image to Base64
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        
        # Clean up to free memory
        del user_img, garment_img, user_img_np, garment_img_np, output_image
        gc.collect()
        torch.cuda.empty_cache()
        
        return {"output_image": output_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

# Define the API endpoint
@app.post("/try-on/", response_model=dict)
async def try_on(request: TryOnRequest):
    result = process_images_standalone(request.user_image_base64, request.garment_image_base64)
    return result

# Root endpoint for health checks
@app.get("/")
async def root():
    return {"message": "Welcome to the Virtual Try-On API. Use POST /try-on/ with base64 image strings.", "status": "healthy"}
