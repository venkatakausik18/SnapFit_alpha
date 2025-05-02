FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

VOLUME /runpod-volume

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]==0.29.0" "aiohttp==3.9.3"

# Build the Quanto CUDA extension
RUN cd /opt/conda/lib/python3.11/site-packages/optimum/quanto/library/extensions/cuda && \
    python setup.py build

COPY . .

EXPOSE 8000

# Verify CUDA is available and Quanto extension
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); import optimum.quanto"

CMD ["python", "-u", "handler.py"]
