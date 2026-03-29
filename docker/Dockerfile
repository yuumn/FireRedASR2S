FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG APP_DIR=/workspace/FireRedASR2S
WORKDIR ${APP_DIR}
COPY . .

ENV PYANNOTE_METRICS_ENABLED=0

RUN export https_proxy=http://10.7.4.2:3128 && \
    export http_proxy=http://10.7.4.2:3128 && \
    apt update && apt-get update && apt install -y ffmpeg && \
    pip install . && \
    pip install omegaconf && \ 
    pip install flask && \
    pip install kaldi_native_fbank && \
    pip install pyannote.audio && \
    unset https_proxy && \
    unset http_proxy

WORKDIR ${APP_DIR}

CMD ["/bin/bash"]