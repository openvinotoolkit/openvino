ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

# Arm libraries are now in ports.ubuntu.com
RUN sed -i 's/^deb /deb [arch=amd64] /g' /etc/apt/sources.list
RUN echo -e "\ndeb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main universe\ndeb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy-updates main universe" >> /etc/apt/sources.list

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install software-properties-common && \
    apt-get update && \
    apt-get install \
        pciutils \
        dmidecode \
        clinfo \
        lshw \
        procps \
        g++ \
        ninja-build \
        smbclient \
        software-properties-common \
        crossbuild-essential-arm64 \
        libpython3-dev:arm64 \
        python3-pip \
        libsqlite3-dev \
        libsqlite3-dev:arm64 \
        zlib1g-dev:arm64 \
        libgflags2.2:arm64 \
        libgflags-dev:arm64 \
        libpugixml1v5:arm64 \
        libtbb2:arm64 \
        nlohmann-json3-dev \
        libpugixml-dev:arm64 \
        libtbb-dev:arm64 \
        shellcheck \
        lintian \
        libffi-dev \
        libssl-dev \
        chrpath \
        libssl-dev \
        libprotobuf-dev \
        libprotoc-dev \
        protobuf-compiler \
        lcov \
        # parallel gzip
        pigz \
        # For ARM CPU plugin
        scons \
        # To extract OpenCV .xz archives
        liblzma-dev \
        # For wheel packaging
        patchelf \
        # check python wheel for duplicated files
        fdupes \
        # Need to curl from pypa without warnings
        openssl \
        # For Java API
        default-jdk \
        # To build documentation
        graphviz \
        texlive \
        liblua5.2-0 \
        # JS API
        libgtk-3-0 \
        libgbm1 \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# For cross-compilation for ARM64. If installed in the first `apt-get install` command, it will be removed by the install_build_dependencies.sh script
RUN apt-get update && \
    apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PIP_VERSION="24.0"
# To cross-compile Python 3.11 we need to first compile it for the host
RUN curl -O https://www.python.org/ftp/python/3.11.3/Python-3.11.3.tar.xz && \
    tar -xf Python-3.11.3.tar.xz && \
    cd Python-3.11.3 && ./configure && make -j4 && make altinstall && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.11 - --no-cache-dir pip==${PIP_VERSION} numpy cython crossenv

# Cross-compile Python 3.11 for ARM64
RUN cd Python-3.11.3 && make distclean && \
    ./configure \
        --host=aarch64-linux-gnu \
        --build=x86_64-linux-gnu \
        --disable-ipv6 \
        --enable-shared \
        --prefix=/opt/python3.11_arm \
        --with-build-python \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j4 && make altinstall

ENV PATH="$SCCACHE_HOME:$PATH"
ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Use Python 3.11 as default
# Using venv here because other methods to switch the default Python break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# Install doxygen
ENV DOXYGEN_HOME="/opt/doxygen"
ENV DOXYGEN_VERSION='1.9.6'
RUN mkdir -p ${DOXYGEN_HOME} && cd ${DOXYGEN_HOME} && wget https://www.doxygen.nl/files/doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz && \
    tar -I pigz -xf doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz && \
    rm -f doxygen-$DOXYGEN_VERSION.linux.bin.tar.gz
ENV PATH="${DOXYGEN_HOME}/doxygen-$DOXYGEN_VERSION/bin:$PATH"
