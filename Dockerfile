FROM nvidia/cuda:10.0-cudnn7-devel
LABEL maintainer=slin@ttic.edu
RUN apt update && apt install -y \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-pip \
    tmux \
    && apt full-upgrade -y && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    matplotlib \
    numpy \
    nvidia-ml-py \
    opencv-python==4.2.0.32 \
    Pillow \
    progressbar2 \
    scikit-learn \
    scipy \
    tensorflow-gpu==1.15
WORKDIR /root
