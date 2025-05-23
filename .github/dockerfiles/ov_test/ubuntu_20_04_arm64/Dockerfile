ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:20.04

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
        ca-certificates \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Python
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        libhdf5-dev \
        # to build samples
        clang \
        && \
    rm -rf /var/lib/apt/lists/*

# Install openvino dependencies
ADD scripts/install_dependencies/install_openvino_dependencies.sh /install_openvino_dependencies.sh
RUN chmod +x /install_openvino_dependencies.sh && \
    /install_openvino_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default instead of Python 3.8
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
ENV PIP_INSTALL_PATH=/venv/lib/python3.11/site-packages

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Change ownership of the venv directory to the non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} /venv

# Switch to the non-root user
USER ${USER_NAME}
