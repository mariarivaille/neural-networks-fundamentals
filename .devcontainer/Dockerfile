FROM python:3.11-slim

ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128


# Install system tools used by the course notebooks and Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    graphviz \
    pkg-config \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install requirements
RUN python -m pip install --upgrade pip setuptools wheel

RUN set -eux; \
    if [ -n "${CUDA_VERSION:-}" ] || [ -d /usr/local/cuda ]; then \
        torch_index_url="$TORCH_CUDA_INDEX_URL"; \
    else \
        torch_index_url="$TORCH_CPU_INDEX_URL"; \
    fi; \
    pip install \
        --default-timeout=1000 \
        --retries=20 \
        --progress-bar off \
        --no-cache-dir \
        --index-url "$torch_index_url" \
        torch==2.7.0 \
        torchvision==0.22.0

RUN grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install \
    --default-timeout=1000 \
    --retries=20 \
    --progress-bar off \
    --no-cache-dir \
    -r /tmp/requirements-no-torch.txt \
    && rm /tmp/requirements-no-torch.txt

# Register jupyter kernel
RUN python3 -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888
