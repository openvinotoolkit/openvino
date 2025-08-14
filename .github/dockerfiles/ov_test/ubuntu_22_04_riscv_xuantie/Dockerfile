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
        git \
        ninja-build \
        ca-certificates \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Compilers
        gcc \
        g++ \
        # qemu build dependencies
        gcc-riscv64-linux-gnu \
        g++-riscv64-linux-gnu \
        libc6-riscv64-cross \
        build-essential \
        pkg-config \
        libglib2.0-dev \
        libpixman-1-dev \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# build xuintie qemu emulator only
ARG XUANTIE_VERSION="V2.8.1"
ARG XUANTIE_REPO="https://github.com/XUANTIE-RV/xuantie-gnu-toolchain"
ARG XUINTIE_PATH="/opt/riscv"
ARG XUINTIE_TMP_PATH="/tmp/xuantie"
ARG XUINTIE_SRC="/tmp/xuantie/src"
    
RUN mkdir -p ${XUINTIE_TMP_PATH} && cd ${XUINTIE_TMP_PATH} && \
    git clone --branch ${XUANTIE_VERSION} --depth 1 ${XUANTIE_REPO} ${XUINTIE_SRC} && \
    cd ${XUINTIE_SRC} && git submodule update --init -- qemu && \
    cd ${XUINTIE_SRC}/qemu && ./configure --prefix=${XUINTIE_PATH} --interp-prefix=/usr/riscv64-linux-gnu --target-list=riscv64-linux-user && \
    make -j$(nproc) && make install && \
    rm -rf ${XUINTIE_TMP_PATH}

# `gh` cli - Download and install GitHub CLI
RUN curl -L https://github.com/cli/cli/releases/download/v2.76.2/gh_2.76.2_linux_amd64.tar.gz -o /tmp/gh.tar.gz && \
    tar -xzf /tmp/gh.tar.gz -C /opt && \
    mv /opt/gh_2.76.2_linux_amd64 /opt/gh && \
    rm /tmp/gh.tar.gz

ENV PATH="/opt/gh/bin:$PATH"
