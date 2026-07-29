# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

# Install RISC-V native debian packages
RUN dpkg --add-architecture riscv64 && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy main restricted > sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy universe >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates universe >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy multiverse >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-updates multiverse >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security main restricted >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security universe >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ jammy-security multiverse >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy main >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy universe >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-updates main >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ jammy-security main >> sources.list && \
    mv sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install software-properties-common wget && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    add-apt-repository --yes --no-update "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/llvm.asc && \
    apt-get update && \
    # install compilers to build OpenVINO for RISC-V 64
    apt-get install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu && \
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
        # For clang-tidy validation
        clang-format-18 \
        clang-tidy-18 \
        libomp-18-dev \
        libncursesw6:riscv64 \
        libtinfo6:riscv64 && \
    apt-get install \
        libpython3-dev:riscv64 && \
    apt-get clean && \
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
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Install newer ninja (apt ships 1.10.1 on 22.04 which lacks --quiet)
RUN python3 -m pip install --no-cache-dir cmake==3.26.4 ninja==1.13.0
