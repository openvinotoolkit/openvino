FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:22.04

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
        # Python \
        python3-dev \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        libxml2-dev \
        libxslt-dev \
        # Compilers
        gcc \
        g++ \
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

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default instead of Python 3.8
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"
