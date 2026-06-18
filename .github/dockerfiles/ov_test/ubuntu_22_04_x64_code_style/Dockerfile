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
    apt-get update && \
    apt-get install \
        curl \
        git \
        ca-certificates \
        cmake \
        wget \
        build-essential \
        shellcheck \
        clang-14 \
        libclang-14-dev \
        && \
    rm -rf /var/lib/apt/lists/*

ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Install custom cmake (4.0+)
RUN wget https://cmake.org/files/v3.31/cmake-3.31.12-linux-x86_64.sh && \
    sh cmake-3.31.12-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.31.12-linux-x86_64.sh
