ARG REGISTRY="docker.io"
FROM ${REGISTRY}/nvidia/cuda:11.8.0-runtime-ubuntu22.04

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
        wget \
        git \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Python
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        # For Java API
        default-jdk \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

# Install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update && apt-get install -y cuda-runtime-11-8 cuda-11-8 && apt-get install -y \
    libcudnn8=8.9.4.*-1+cuda11.8 \
    libcudnn8-dev=8.9.4.*-1+cuda11.8 \
    libcudnn8-samples=8.9.4.*-1+cuda11.8 \
    libcutensor1=1.7.0.1-1 \
    libcutensor-dev=1.7.0.1-1 \
    cuda-drivers

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default
# Using venv here 'cause other methods to switch the default Python on Ubuntu 22 break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
RUN mkdir -p /home/openvino && chown -R ${USER_NAME}:${USER_NAME} /home/openvino

# Change ownership of the venv directory to the non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} /venv

# Switch to the non-root user
USER ${USER_NAME}