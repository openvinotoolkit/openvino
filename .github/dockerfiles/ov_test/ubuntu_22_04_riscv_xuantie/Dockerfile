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
        build-essential \
        pkg-config \
        libglib2.0-dev \
        libpixman-1-dev \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# build xuintie qemu emulator
ARG QEMU_VERSION="5af674035298c07b361de69afdabf2fcbb37da71"
ARG QEMU_REPO="https://github.com/XUANTIE-RV/qemu"
ARG QEMU_PATH="/opt/riscv"
ARG QEMU_TMP_PATH="/tmp/qemu"
ARG QEMU_SRC="/tmp/qemu/src"
    
RUN mkdir -p ${QEMU_TMP_PATH} && cd ${QEMU_TMP_PATH} && \
    git clone --branch ${QEMU_VERSION} --depth 1 ${QEMU_REPO} ${QEMU_SRC} && cd ${QEMU_SRC} && \
    ./configure --prefix=${QEMU_PATH} --target-list=riscv64-linux-user && \
    make -j$(nproc) && make install && \
    rm -rf ${QEMU_TMP_PATH}
