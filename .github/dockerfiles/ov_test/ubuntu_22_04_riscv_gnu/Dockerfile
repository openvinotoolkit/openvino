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
        libglib2.0-dev \
        libpixman-1-dev \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

ARG QEMU_VERSION="10.2.0"

RUN mkdir -p /tmp/qemu-src && cd /tmp/qemu-src && \
    curl -SLO https://github.com/qemu/qemu/archive/refs/tags/v${QEMU_VERSION}.tar.gz && \
    tar -xzf v${QEMU_VERSION}.tar.gz --strip-components=1 && \
    ./configure --target-list=riscv64-linux-user --prefix=/usr/local && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /tmp/qemu-src
