ARG REGISTRY="quay.io"
FROM openvinogithubactions.azurecr.io/quayio/pypa/manylinux2014_x86_64

USER root

# Create a non-root user and group
ENV USER_NAME=runner
ENV USER_ID=1000
ENV GROUP_NAME=${USER_NAME}
ENV GROUP_ID=${USER_ID}

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && bash -e /install_build_dependencies.sh

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Switch to the non-root user
USER ${USER_NAME}
