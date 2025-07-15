FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST_VER="6.0;7.0;7.5;8.0;8.6"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    make \
    cmake \
    ninja-build \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    libopenblas-dev \
    libgl1-mesa-glx \
    gcc \
    g++ \
    llvm \
    gawk \
    curl \
    ca-certificates \
    libomp-dev \
    tzdata \
    python3-pycuda \
    ninja-build \
    xterm \
    xauth \
    openssh-server \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Copy the current directory and set is as the work directory
COPY . /GrowSP-ForMS
WORKDIR /GrowSP-ForMS

# Install miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create conda environment from file
RUN conda env create -f env.yaml
ARG CONDA_ENV_NAME=growsp_forms
ENV PATH=$CONDA_DIR/envs/$CONDA_ENV_NAME/bin:$PATH
ENV CONDA_DEFAULT_ENV=$CONDA_ENV_NAME
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install Minkowski Engine
RUN git clone --recursive https://github.com/NVIDIA/MinkowskiEngine /workspaces/MinkowskiEngine
RUN bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV_NAME && \
    export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST_VER}\" && \
    cd /workspaces/MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas"

# Build cutpursuit
COPY ./preprocess/cut_pursuit /GrowSP-ForMS/preprocess/cut_pursuit
RUN bash -c "source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $CONDA_ENV_NAME && \
    cd /GrowSP-ForMS/preprocess/cut_pursuit && ./install.sh"

# Initialize Conda for bash
RUN $CONDA_DIR/bin/conda init bash
# Auto-activate the environment in new shells
RUN echo "conda activate ${CONDA_DEFAULT_ENV}" >> ~/.bashrc

CMD ["bash"]