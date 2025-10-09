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

RUN apt update && \
    apt install wget git curl cmake lsb-release software-properties-common gnupg build-essential && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 18 && \
    apt --assume-yes install binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu g++-aarch64-linux-gnu scons gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu
