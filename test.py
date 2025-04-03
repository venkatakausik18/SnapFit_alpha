import torch
import numpy as np
from PIL import Image
import os
import base64
import io
from src.pipeline_tryon import FluxTryonPipeline, resize_by_height
from transformers import T5EncoderModel, CLIPTextModel
from diffusers import FluxTransformer2DModel, AutoencoderKL

# Device and dtype setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.bfloat16

def load_models(device=device, torch_dtype=torch_dtype):
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=torch_dtype)
    text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=torch_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch_dtype)
    vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae")
    
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
        model_image = resize_by_height(model_image, height)
        concat_image_list.append(model_image)
    if has_garment_image:
        garment_image = resize_by_height(garment_image, height)
        concat_image_list.append(garment_image)
    
    image = np.concatenate([np.array(img) for img in concat_image_list], axis=1)
    image = Image.fromarray(image)
    
    mask = np.zeros_like(np.array(image))
    mask[:, :width] = 255
    mask_image = Image.fromarray(mask)
    
    output = pipe(
        "",  # Empty prompt
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
    """
    Process user and garment images for virtual try-on.
    
    Args:
        user_image_base64: Base64 encoded string of the user image
        garment_image_base64: Base64 encoded string of the garment image
        
    Returns:
        Dictionary containing base64 encoded string of the output image
        
    Raises:
        Exception: If any error occurs during processing
    """
    try:
        # Decode Base64 strings to images
        user_image_data = base64.b64decode(user_image_base64)
        garment_image_data = base64.b64decode(garment_image_base64)
        
        user_img = Image.open(io.BytesIO(user_image_data)).convert("RGBA")
        garment_img = Image.open(io.BytesIO(garment_image_data)).convert("RGBA")
        
        # Convert to NumPy arrays
        user_img_np = np.array(user_img)
        garment_img_np = np.array(garment_img)
        
        # Generate the try-on image
        output_image = generate_image(model_image=user_img_np, garment_image=garment_img_np)
        
        # Convert output image to Base64
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        
        return {"output_image": output_base64}
    except Exception as e:
        raise Exception(f"Error processing images: {str(e)}")

# Main block to handle file inputs
if __name__ == "__main__":
    # Specify your input and output file paths here
    user_image_path = "asset/images/model/model1.png"  # Replace with your actual user image path
    garment_image_path = "asset/images/garment/garment1.jpg"  # Replace with your actual garment image path
    output_path = "asset/images/model/model1.png"  # Replace with your desired output path

    try:
        # Read the image files and convert to base64
        with open(user_image_path, "rb") as f:
            user_image_base64 = base64.b64encode(f.read()).decode("utf-8")
        with open(garment_image_path, "rb") as f:
            garment_image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Call the existing function with base64 inputs
        result = process_images_standalone(user_image_base64, garment_image_base64)

        # Decode the output base64 and save to file
        output_bytes = base64.b64decode(result["output_image"])
        with open(output_path, "wb") as f:
            f.write(output_bytes)
        print(f"Success! Output saved to {output_path}")
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
