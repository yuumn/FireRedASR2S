FROM 

ARG APP_DIR=/workspace/FireRedASR2S
WORKDIR ${APP_DIR}
COPY . .

ENV PYANNOTE_METRICS_ENABLED=0
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN export https_proxy=http://10.7.4.2:3128 && \
    export http_proxy=http://10.7.4.2:3128 && \
    python -m pip install peft && \
    apt-get clean && rm -rf /var/lib/apt/lists/* \
    unset https_proxy && \
    unset http_proxy

WORKDIR ${APP_DIR}

CMD ["/bin/bash"]