# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

# Install RISC-V native debian packages
RUN dpkg --add-architecture riscv64 && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble main restricted > sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble-updates main restricted >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble universe >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble-updates universe >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble multiverse >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble-updates multiverse >> sources.list && \
    echo deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ noble-backports main restricted universe multiverse >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ noble-security main restricted >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ noble-security universe >> sources.list && \
    echo deb [arch=amd64] http://security.ubuntu.com/ubuntu/ noble-security multiverse >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ noble main >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ noble universe >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ noble-updates main >> sources.list && \
    echo deb [arch=riscv64] http://ports.ubuntu.com/ubuntu-ports/ noble-security main >> sources.list && \
    mv sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
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
    procps \
    # parallel gzip
    pigz \
    # Python \
    python3-dev \
    python3-pip \
    python3-venv \
    # Compilers
    gcc \
    g++ \
    clang-format-18 \
    clang-tidy-18 \
    libomp-18-dev && \
    # Pin ncurses/tinfo to the noble release-pocket version for both arches. The
    # amd64 and riscv64 (ports) mirrors update these multiarch "same" packages on
    # different schedules, so an updated version is often available for only one
    # arch, causing a "Breaks" version conflict. The release-pocket version is the
    # one version guaranteed to exist on both mirrors permanently.
    NCURSES_VERSION=$(apt-cache madison libtinfo6 | awk -F'|' '$3 ~ / noble\/main / {gsub(/ /, "", $2); print $2; exit}') && \
    apt-get install --allow-downgrades \
    libncurses6=${NCURSES_VERSION} \
    libncursesw6=${NCURSES_VERSION} \
    libtinfo6=${NCURSES_VERSION} \
    libncurses6:riscv64=${NCURSES_VERSION} \
    libncursesw6:riscv64=${NCURSES_VERSION} \
    libtinfo6:riscv64=${NCURSES_VERSION} && \
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

RUN python3 -m venv venv
ENV PATH="/venv/bin:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

RUN python3 -m pip install --no-cache-dir cmake==3.26.4 ninja==1.13.0
