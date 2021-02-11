FROM nvidia/cuda:10.0-cudnn7-devel
LABEL maintainer=slin@ttic.edu
ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y \
    cmake \
    git \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    mesa-common-dev \
    python-pip \
    python-tk \
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
RUN git clone --depth 1 https://github.com/YoYo000/fusibile && \
    cd fusibile && \
    cmake . && \
    make -j
