# Thin image: Python + the GL libraries MuJoCo needs + uv. Torch brings its own CUDA
# runtime, so there is no CUDA base image here — only the host driver is required, and
# only if you want GPU acceleration at all.
FROM python:3.12-slim-bookworm

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libosmesa6 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libxrandr2 \
        libxinerama1 \
        libxcursor1 \
        libxi6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

USER ${USERNAME}
WORKDIR /workspace

# The virtualenv lives inside the bind-mounted workspace so it inherits the host user's
# ownership automatically and survives image rebuilds. Anything created inside the image
# instead (a home directory, a named cache volume) is owned by uid 1000, which breaks as
# soon as the host uid differs — and on a domain-joined machine it always does.
ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/workspace/.venv/bin:${PATH} \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
