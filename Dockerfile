FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common

RUN add-apt-repository ppa:savoury1/ffmpeg4

RUN apt-get install -y --no-install-recommends pkg-config
RUN apt install -y --no-install-recommends ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev

RUN pip install --upgrade setuptools
# Lock specific opencv version to avoid issues with GStreamerPipeline
RUN pip install --upgrade opencv-python==4.5.5.64
RUN pip install \
    cython \
    pillow \
    pycocotools \
    matplotlib \
    wheel \
    pypng \
    'av>=7.0.1' \
    decord \
    nuscenes-devkit \
    ifcfg \
    einops \
    wandb
