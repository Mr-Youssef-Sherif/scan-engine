FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python, pip, and system dependencies
RUN apt update && apt install -y \
    ffmpeg \
    curl \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure python and pip commands are always available
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install requirements first to leverage caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . /app
WORKDIR /app

# RunPod doesn't need CMD or ENTRYPOINT — it uses handler.py's handler() function
