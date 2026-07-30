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
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

USER ${USERNAME}
WORKDIR /workspace

# The virtualenv deliberately lives outside /workspace: that directory is bind-mounted
# from the host, so a .venv inside it would be shadowed by (or leak into) the host's.
ENV UV_PROJECT_ENVIRONMENT=/home/${USERNAME}/.venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/home/${USERNAME}/.venv/bin:${PATH} \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
