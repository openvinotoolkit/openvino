ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/debian:10.13

USER root

# APT configuration
# WARNING: Debian 10 "Buster" is no longer officially supported,
# so repositories were moved to "archive.debian.org"
# See: https://www.debian.org/releases/
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf && \
    sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i '/stretch-updates/d' /etc/apt/sources.list

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install  \
        git \
        libc6-dev \
        # parallel gzip
        pigz \
        # Python for running C++ Functional tests via parallelization script  \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        # Compilers
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/arm-linux-gnueabihf/:/usr/arm-linux-gnueabihf/lib/:$LD_LIBRARY_PATH"
RUN ln -s /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 /lib/ld-linux-armhf.so.3
