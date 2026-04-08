FROM artifactory.devops.xiaohongshu.com/dockerhub/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG APP_DIR=/workspace/FireRedASR2S
WORKDIR ${APP_DIR}
COPY . .

ENV PYANNOTE_METRICS_ENABLED=0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN export https_proxy=http://10.7.4.2:3128 && \
    export http_proxy=http://10.7.4.2:3128 && \
    apt update && apt-get update && apt-get install -y vim ffmpeg python3.12 python3.12-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python -m pip install . && \
    python -m pip install ffmpeg-python && \
    python -m pip install omegaconf && \
    python -m pip install flask && \
    python -m pip install kaldi_native_fbank && \
    python -m pip install pyannote.audio gunicorn gevent nvitop && \
    python -m pip install torchcodec==0.10 && \
    python -m pip install peft && \
    python -m pip install flash-attn --no-build-isolation && \
    python -m pip install httpx urllib3 filetype && \
    apt-get clean && rm -rf /var/lib/apt/lists/* \
    unset https_proxy && \
    unset http_proxy

WORKDIR ${APP_DIR}

CMD ["/bin/bash"]