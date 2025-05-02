FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

VOLUME /runpod-volume

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]==0.29.0" "aiohttp==3.9.3"

# Check if the quanto directory exists and what files it contains
RUN if [ -d "/opt/conda/lib/python3.11/site-packages/optimum/quanto" ]; then \
    ls -la /opt/conda/lib/python3.11/site-packages/optimum/quanto; \
    if [ -d "/opt/conda/lib/python3.11/site-packages/optimum/quanto/library/extensions/cuda" ]; then \
    ls -la /opt/conda/lib/python3.11/site-packages/optimum/quanto/library/extensions/cuda; \
    fi; \
    fi

# Install optimum with CUDA support
RUN pip install --no-cache-dir optimum[cuda]

COPY . .

EXPOSE 8000

# Verify CUDA is available
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); import optimum.quanto"

CMD ["python", "-u", "handler.py"]
