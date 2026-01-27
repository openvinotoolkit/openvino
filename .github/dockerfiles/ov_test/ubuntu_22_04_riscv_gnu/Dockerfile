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
        ca-certificates \
        curl \
        git \
        gpg-agent \
        ninja-build \
        tzdata \
        # parallel gzip
        pigz \
        # Compilers
        gcc \
        g++ \
        gcc-riscv64-linux-gnu \
        g++-riscv64-linux-gnu \
        binutils-riscv64-linux-gnu \
        libc6-riscv64-cross \
        libstdc++6-riscv64-cross \
        libgcc-s1-riscv64-cross \
        qemu-user \
        # QEMU build deps
        build-essential \
        meson \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        python3-tomli \
        autoconf \
        automake \
        bison \
        flex \
        gawk \
        gperf \
        libexpat-dev \
        libgmp-dev \
        libmpc-dev \
        libmpfr-dev \
        libtool \
        texinfo \
        libglib2.0-dev \
        libpixman-1-dev \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

ARG QEMU_VERSION="10.2.0"

# Build QEMU from riscv-gnu-toolchain to get recent RVV support
RUN mkdir -p /tmp/riscv-gnu-toolchain && cd /tmp/riscv-gnu-toolchain && \
    git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git && \
    cd riscv-gnu-toolchain && \
    git checkout 2026.01.23 && \
    ./configure --prefix=/opt/riscv && \
    make -j"$(nproc)" && \
    make -j"$(nproc)" build-qemu && \
    rm -rf /tmp/riscv-gnu-toolchain
