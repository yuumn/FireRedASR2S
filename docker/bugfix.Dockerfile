FROM artifactory.devops.xiaohongshu.com/media/fireredasr2s:260407-red9

ARG APP_DIR=/workspace/FireRedASR2S
WORKDIR ${APP_DIR}
COPY . .

ENV PYANNOTE_METRICS_ENABLED=0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN export https_proxy=http://10.7.4.2:3128 && \
    export http_proxy=http://10.7.4.2:3128 && \
    python -m pip install httpx urllib3 filetype && \
    unset https_proxy && \
    unset http_proxy

WORKDIR ${APP_DIR}

CMD ["/bin/bash"]