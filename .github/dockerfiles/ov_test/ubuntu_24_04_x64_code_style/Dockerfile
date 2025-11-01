ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:24.04

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
    apt-get update && \
    apt-get install \
        curl \
        git \
        ca-certificates \
        wget \
        cmake \
        lsb-release \
        gnupg \
        build-essential \
        clang-format-18 \
        shellcheck \
        python3-dev \
        python3-setuptools \
        python3-pip \
        && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 && \
    apt-get --assume-yes install binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu g++-aarch64-linux-gnu scons gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu && \
    rm -rf /var/lib/apt/lists/*
