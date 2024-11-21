ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:20.04

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
    ## 1 patch
    # install compilers to build OpenVINO for RISC-V 64
    apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu && \
    ##
    apt-get install \
        curl \
        git \
        ## 2 patch
        cmake \
        ccache \
        ninja-build \
        fdupes \
        patchelf \
        ca-certificates \
        ##
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        ## patch 3
        # Python \
        # Pythons
        python3-dev \
        python3.9-dev \
        python3-pip \
        python3-venv \
        python3.9-venv \
        python3-distutils \
        python3.9-distutils \
        # Compilers
        # For Java API
        gcc \
        default-jdk \
        g++ \
        ##
        ## patch 4
        # # Pythons
        # python3.9-dev \
        # python3.9-venv \
        # python3.9-distutils \
        # # For Java API
        # default-jdk \
        ##
        && \
    rm -rf /var/lib/apt/lists/*
## patch 5
# Install RISC-V native debian packages
# RUN echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy main restricted > riscv64-sources.list

RUN echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy main restricted > riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security main restricted >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security universe >> riscv64-sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security multiverse >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy main >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy universe >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main >> riscv64-sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> riscv64-sources.list \
    mv riscv64-sources.list /etc/apt/sources.list.d/

##
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

ENV PATH="$SCCACHE_HOME:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py  --no-cache-dir pip==${PIP_VERSION} && \
    python3.9 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.9 as default instead of Python 3.8
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.9 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
