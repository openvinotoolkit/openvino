ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/debian:10.13

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install \
        software-properties-common \
        curl \
        git \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Pythons \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        libhdf5-dev \
        # For building Python from source
        build-essential \
        libffi-dev \
        libgdbm-dev \
        libc6-dev \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        liblzma-dev \
        wget \
        # Compilers
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-aarch64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Setup Python
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz

RUN tar -xf Python-3.11.9.tar.xz && \
    cd Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall

# Setup pip
ENV PIP_VERSION="24.0"
RUN python3 -m pip install --upgrade pip==24.0
RUN python3.11 -m pip install --upgrade pip==24.0

# Use Python 3.11 as default
# Using venv here because other methods to switch the default Python break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
