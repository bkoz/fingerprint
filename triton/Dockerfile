FROM nvcr.io/nvidia/tritonserver:22.11-py3
LABEL maintainer="Bob Kozdemba <bkozdemba@gmail.com>"

### Setup user for build execution and application runtime
ENV APP_ROOT=/opt/app-root
RUN mkdir -p ${APP_ROOT}/{bin,src} && \
    chmod -R u+x ${APP_ROOT}/bin && chgrp -R 0 ${APP_ROOT} && chmod -R g=u ${APP_ROOT}
ENV PATH=${APP_ROOT}/bin:${PATH} HOME=${APP_ROOT}

WORKDIR ${APP_ROOT}/src
COPY . ${APP_ROOT}/src

### Containers should NOT run as root as a good practice
USER 1001

EXPOSE 8000 8001 8002

CMD tritonserver --model-repository=${MODEL_REPOSITORY}

