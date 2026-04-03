FROM artifactory.devops.xiaohongshu.com/media/fireredasr2s:260401-red4

ARG APP_DIR=/workspace/FireRedASR2S
WORKDIR ${APP_DIR}
COPY . .

ENV PYANNOTE_METRICS_ENABLED=0
ENV PIP_BREAK_SYSTEM_PACKAGES=1


WORKDIR ${APP_DIR}

CMD ["/bin/bash"]