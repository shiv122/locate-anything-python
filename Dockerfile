# LocateAnything service: FastAPI (URL-based API) loading NVIDIA's
# LocateAnything-3B visual-grounding model via transformers. Weights + custom
# trust_remote_code files are baked in at build time so cold starts don't pull
# from the Hub. Needs a GPU at runtime (nvidia-container-toolkit / a GPU host on
# Coolify).
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

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

# Torch matched to the CUDA 12.1 base. The deploy host (RTX 3060) maxes out at
# CUDA 12.2, so we stay at/below it — a cu124 build hit CUDA error 804 there
# (GeForce cards can't use forward compatibility).
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121

# Python deps (cached unless requirements change).
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the model (weights + trust_remote_code files) into the image. The repo is
# public, so no token is needed. If you ever point LOCATE_MODEL at a GATED repo,
# don't use ARG/ENV for the token — mount it as a BuildKit secret instead:
#   RUN --mount=type=secret,id=hf_token \
#       HF_TOKEN="$(cat /run/secrets/hf_token)" python3 -c "..."
#   docker buildx build --secret id=hf_token,src=./hf_token.txt ...
RUN python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ.get('LOCATE_MODEL', 'nvidia/LocateAnything-3B'))"

# GeForce cards (e.g. RTX 3060) don't support CUDA "forward compatibility", but
# the base image's cuda-compat libs try to use it and fail with CUDA error 804
# ("forward compatibility was attempted on non supported HW"). Drop them so the
# container uses the host's own driver (libcuda) injected by the NVIDIA runtime.
# Kept as a late layer so the cached torch + model layers above don't rebuild.
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda-*/compat && ldconfig

COPY app ./app
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
