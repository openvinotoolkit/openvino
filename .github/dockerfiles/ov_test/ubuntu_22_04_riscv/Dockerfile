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
        # Python
        python3 \
        python3-tomli \
        python3-venv \
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

# build riscv-collab qemu emulator only
ARG RISCV_GNU_TOOLCHAIN_REF="2026.05.06"
ARG RISCV_GNU_TOOLCHAIN_REPO="https://github.com/riscv-collab/riscv-gnu-toolchain.git"
ARG RISCV_TOOLCHAIN_PATH="/opt/riscv"
ARG RISCV_TOOLCHAIN_TMP_PATH="/tmp/riscv-gnu-toolchain"
ARG RISCV_TOOLCHAIN_SRC="/tmp/riscv-gnu-toolchain/src"

RUN mkdir -p ${RISCV_TOOLCHAIN_TMP_PATH} && cd ${RISCV_TOOLCHAIN_TMP_PATH} && \
    git clone --branch ${RISCV_GNU_TOOLCHAIN_REF} --depth 1 ${RISCV_GNU_TOOLCHAIN_REPO} ${RISCV_TOOLCHAIN_SRC} && \
    cd ${RISCV_TOOLCHAIN_SRC} && git submodule update --init -- qemu && \
    cd ${RISCV_TOOLCHAIN_SRC}/qemu && \
    ./configure --prefix=${RISCV_TOOLCHAIN_PATH} --interp-prefix=/usr/riscv64-linux-gnu --target-list=riscv64-linux-user && \
    make -j"$(nproc)" && make install && \
    rm -rf ${RISCV_TOOLCHAIN_TMP_PATH}
