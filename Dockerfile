FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    git libgl1 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA installation path
RUN ls -la ${CUDA_HOME} || echo "CUDA path not found"

VOLUME /runpod-volume

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]==0.29.0" "aiohttp==3.9.3"

# Install optimum with CUDA support
RUN pip install --no-cache-dir optimum[cuda]

COPY . .

EXPOSE 8000

# Verify CUDA is available and environment variables are set
RUN python -c "import torch; import os; print('CUDA available:', torch.cuda.is_available()); print('CUDA_HOME:', os.environ.get('CUDA_HOME')); import optimum.quanto"

CMD ["python", "-u", "handler.py"]
