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

# Install dependencies
RUN pip install --no-cache-dir huggingface_hub transformers diffusers torch torchvision torchaudio

# Set model cache directory
ENV MODEL_CACHE_DIR=/models

# Create model directory
RUN mkdir -p $MODEL_CACHE_DIR

# Download models manually using wget
RUN wget -P $MODEL_CACHE_DIR https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/pytorch_model.bin
RUN wget -P $MODEL_CACHE_DIR https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/pytorch_model.bin
RUN wget -P $MODEL_CACHE_DIR https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/transformer/pytorch_model.bin
RUN wget -P $MODEL_CACHE_DIR https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/pytorch_model.bin


# Copy the entire project
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
