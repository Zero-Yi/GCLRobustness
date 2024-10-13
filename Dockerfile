# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for Python installation
ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA to install python3.9 and related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    wget \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# Install PyTorch 2.1 with CUDA 11.8
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install DGL (Deep Graph Library) for PyTorch 2.1 and CUDA 11.8
RUN pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

# Install PyTorch Geometric and dependencies (compatible with PyTorch 2.1 and CUDA 11.8)
RUN pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install Scikit-learn and SciPy
RUN pip install scikit-learn scipy

# Install PyGCL
RUN pip install PyGCL

# Set the working directory
WORKDIR /workspace

# Verify installation
RUN python3 -c "import torch; print(torch.cuda.is_available())" \
    && python3 -c "import dgl; print(dgl.__version__)" \
    && python3 -c "import torch_geometric; print(torch_geometric.__version__)"

# The image will launch with bash
CMD ["/bin/bash"]
