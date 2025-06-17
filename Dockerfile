# Base CUDA 11.6 development image with Ubuntu 20.04
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9

# Install Python 3.9, core libraries, and GPU tools
RUN apt-get update && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
        build-essential \
        g++ \
        curl \
        unzip \
        git \
        tzdata \
        ninja-build \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        sudo && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    pip install --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: install NodeJS + npx
RUN apt-get update && apt-get install -y nodejs npm && npm install -g npx

# Install Jupyter
RUN pip install notebook ipykernel

# Set working directory
WORKDIR /workspace

# Optionally copy your codebase into container
# COPY . .

# Expose port for Jupyter
EXPOSE 5018

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=5018", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

