FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Set CUDA environment variables explicitly
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    git libgl1 \
    && rm -rf /var/lib/apt/lists/*

VOLUME /runpod-volume

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "uvicorn[standard]==0.29.0" "aiohttp==3.9.3"

COPY . .

EXPOSE 8000

# Verify CUDA is available (optional debugging step)
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "-u", "handler.py"]
