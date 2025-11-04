# Use official PyTorch image as base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

# Create directories for data and outputs
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/samples

# Default command (can be overridden)
CMD ["python", "--version"]

