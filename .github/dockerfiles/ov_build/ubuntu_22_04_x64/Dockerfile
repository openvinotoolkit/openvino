ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

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
        libtbb2 \
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
        # To build documentation
        graphviz \
        texlive \
        liblua5.2-0 \
        # JS API
        xvfb \
        libgtk-3-0 \
        libgbm1 \
        # ONNX Runtime
        language-pack-en \
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

# ONNX Runtime, see https://github.com/microsoft/onnxruntime/issues/13197#issuecomment-1264542497
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# Install doxygen
ENV DOXYGEN_HOME="/opt/doxygen"
ENV DOXYGEN_VERSION='1.9.6'
RUN mkdir -p ${DOXYGEN_HOME} && cd ${DOXYGEN_HOME} && wget https://www.doxygen.nl/files/doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz && \
    tar -I pigz -xf doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz && \
    rm -f doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz
ENV PATH="${DOXYGEN_HOME}/doxygen-$DOXYGEN_VERSION/bin:$PATH"

# `gh` CLI
RUN mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y \
    && rm -rf /var/lib/apt/lists/*
