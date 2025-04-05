# Base image with PyTorch + CUDA
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy your local code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    fastapi \
    uvicorn[standard] \
    numpy \
    pillow \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    xformers \
    huggingface_hub \
    tqdm \
    opencv-python \
    scikit-image \
    einops \
    peft

# Expose API port
EXPOSE 8000

# Set environment variable to prevent token warnings
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Ensure model is loaded from volume
ENV MODEL_DIR=/workspace/checkpoints

# Launch the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
