ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# Create a non-root user and group
ENV USER_NAME=runner
ENV USER_ID=1000
ENV GROUP_NAME=${USER_NAME}
ENV GROUP_ID=${USER_ID}

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install \
        curl \
        git \
        gpg-agent \
        tzdata \
        libtbb2 \
        # ONNX Runtime build
        language-pack-en \
        # parallel gzip
        pigz \
        # Pythons
        python3.9-dev \
        python3.9-venv \
        python3.9-distutils \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        python3.12-dev \
        python3.12-venv \
        python3.13-dev \
        python3.13-venv \
        # For Java API
        default-jdk \
        # OpenVINO JS API
        libgtk-3-0 \
        libgbm1 \
        xvfb \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# ONNX Runtime, see https://github.com/microsoft/onnxruntime/issues/13197#issuecomment-1264542497
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py  --no-cache-dir pip==${PIP_VERSION} && \
    python3.9 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.10 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.12 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.13 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default
# Using venv here 'cause other methods to switch the default Python on Ubuntu break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
RUN mkdir -p /home/openvino && chown -R ${USER_NAME}:${USER_NAME} /home/openvino

# Change ownership of the venv directory to the non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} /venv

# Change ownership of node to the non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} ${NVM_DIR}

# Switch to the non-root user
USER ${USER_NAME}

