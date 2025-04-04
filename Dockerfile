# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip show runpod

## Set environment variable for model cache
ENV MODEL_CACHE_DIR=/models/flux_tryon

# Install dependencies
RUN pip install huggingface_hub transformers diffusers torch torchvision torchaudio 

# Preload models into cache (avoiding downloads at runtime)
RUN python -c "from transformers import CLIPTextModel, T5EncoderModel; \
    from diffusers import FluxTransformer2DModel, AutoencoderKL; \
    CLIPTextModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='text_encoder', cache_dir='$MODEL_CACHE_DIR'); \
    T5EncoderModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='text_encoder_2', cache_dir='$MODEL_CACHE_DIR'); \
    FluxTransformer2DModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='transformer', cache_dir='$MODEL_CACHE_DIR'); \
    AutoencoderKL.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='vae', cache_dir='$MODEL_CACHE_DIR')"

# Copy the entire project
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
