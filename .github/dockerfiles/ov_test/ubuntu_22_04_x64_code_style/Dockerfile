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

RUN apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install \
        curl \
        git \
        ca-certificates \
        wget \
        build-essential \
        shellcheck \
        clang-14 \
        libclang-14-dev \
        python3.11-dev \
        python3.11-venv \
        && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.11 as default instead of Python 3.10
# Using venv here 'cause other methods to switch the default Python on Ubuntu break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"

# 'packaging' is required by OpenVINO cmake (check_python_requirements.py). get-pip.py installs it
# for the system Python 3.11, but not as part of the venv
# chown: code_style jobs run the container as uid 1001, so the venv must be writable for pip install, see
# https://github.com/openvinotoolkit/openvino/blob/154c1eecbad6a481cae9a110cd28dbe6842f2322/.github/workflows/code_style.yml#L210
RUN python3 -m pip install --no-cache-dir packaging && \
    chown -R 1001 /venv

# Install CMake 3.26.4 using the Kitware installer script
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    sh cmake-3.26.4-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.26.4-linux-x86_64.sh
