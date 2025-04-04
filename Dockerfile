# Use official PyTorch image with CUDA 12.4
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Set model cache directory
ENV MODEL_CACHE_DIR=/models/flux_tryon
RUN mkdir -p $MODEL_CACHE_DIR

# Use curl instead of wget with retry logic
RUN curl --retry 5 -L -o $MODEL_CACHE_DIR/text_encoder.bin \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder/pytorch_model.bin && \
    curl --retry 5 -L -o $MODEL_CACHE_DIR/text_encoder_2.bin \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/text_encoder_2/pytorch_model.bin && \
    curl --retry 5 -L -o $MODEL_CACHE_DIR/transformer.bin \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/transformer/pytorch_model.bin && \
    curl --retry 5 -L -o $MODEL_CACHE_DIR/vae.bin \
    https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/pytorch_model.bin

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
