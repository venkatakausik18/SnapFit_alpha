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
# Cache the model inside the image
RUN mkdir -p /models/flux_tryon
RUN pip install huggingface_hub && \
    python -c "from transformers import CLIPTextModel, T5EncoderModel; from diffusers import FluxTransformer2DModel, AutoencoderKL; \
               CLIPTextModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='text_encoder', cache_dir='/models/flux_tryon'); \
               T5EncoderModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='text_encoder_2', cache_dir='/models/flux_tryon'); \
               FluxTransformer2DModel.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='transformer', cache_dir='/models/flux_tryon'); \
               AutoencoderKL.from_pretrained('black-forest-labs/FLUX.1-dev', subfolder='vae', cache_dir='/models/flux_tryon')"

# Copy the entire project
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
