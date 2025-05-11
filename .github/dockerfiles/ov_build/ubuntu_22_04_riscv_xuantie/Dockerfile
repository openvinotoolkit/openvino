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
    # install compilers to build OpenVINO for RISC-V 64
    apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu && \
    apt-get install \
        curl \
        git \
        cmake \
        ccache \
        ninja-build \
        fdupes \
        patchelf \
        ca-certificates \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Python \
        python3-dev \
        python3-pip \
        python3-venv \
        python3-distutils \
        # Compilers
        gcc \
        g++ \
        # xuantie-gnu-toolchain build dependencies
        autoconf \
        automake \
        autotools-dev \
        libmpc-dev \
        libmpfr-dev\ 
        libgmp-dev \
        gawk \
        build-essential \
        bison \
        flex \
        texinfo \
        gperf \
        libtool \
        patchutils \
        bc \
        zlib1g-dev \
        libexpat-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install RISC-V native debian packages
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
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> riscv64-sources.list && \
    mv riscv64-sources.list /etc/apt/sources.list.d/

RUN dpkg --add-architecture riscv64 && \
    apt-get update -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/riscv64-sources.list && \
    apt-get install -y --no-install-recommends libpython3-dev:riscv64

# build xuintie toolchain
ARG XUANTIE_VERSION="V2.8.1"
ARG XUANTIE_REPO="https://github.com/XUANTIE-RV/xuantie-gnu-toolchain"
ARG XUINTIE_PATH="/opt/riscv"
ARG XUINTIE_TMP_PATH="/tmp/xuantie"
ARG XUINTIE_SRC="/tmp/xuantie/src"
    
RUN mkdir -p ${XUINTIE_TMP_PATH} && cd ${XUINTIE_TMP_PATH} && \
    git clone --branch ${XUANTIE_VERSION} --depth 1 ${XUANTIE_REPO} ${XUINTIE_SRC} && cd ${XUINTIE_SRC} && \
    ./configure --prefix=${XUINTIE_PATH} --disable-gdb && \
    make linux -j$(nproc) && make install && \
    rm -rf ${XUINTIE_TMP_PATH}

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py
