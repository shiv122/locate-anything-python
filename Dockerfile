# LocateAnything service: FastAPI (URL-based API) loading NVIDIA's
# LocateAnything-3B visual-grounding model via transformers. Weights + custom
# trust_remote_code files are baked in at build time so cold starts don't pull
# from the Hub. Needs a GPU at runtime (nvidia-container-toolkit / a GPU host on
# Coolify).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    LOCATE_MODEL=nvidia/LocateAnything-3B \
    LOCATE_DEVICE=cuda \
    GENERATION_MODE=hybrid \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tini build-essential \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch matched to the CUDA 12.4 base (installed separately per the model card).
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Python deps (cached unless requirements change).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the model (weights + trust_remote_code files) into the image. Pass a
# Hugging Face token at build time if the repo is gated:
#   docker build --build-arg HF_TOKEN=hf_xxx -t locate-anything-service .
ARG HF_TOKEN=
RUN HF_TOKEN="${HF_TOKEN}" python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ.get('LOCATE_MODEL', 'nvidia/LocateAnything-3B'), token=(os.environ.get('HF_TOKEN') or None))"

COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
