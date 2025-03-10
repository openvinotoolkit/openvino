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
    apt-get install  \
        git \
        libc6-dev \
        # parallel gzip
        pigz \
        # Python
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        # To build Python 3.10 from source
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
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# Install openvino dependencies
ADD scripts/install_dependencies/install_openvino_dependencies.sh /install_openvino_dependencies.sh
RUN chmod +x /install_openvino_dependencies.sh && \
    /install_openvino_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Setup Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tar.xz

RUN tar -xf Python-3.10.9.tar.xz && \
    cd Python-3.10.9 && \
    ./configure --enable-optimizations && \
    make -j 8 && \
    make altinstall

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.10 as default instead of Python 3.7
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.10 -m venv venv
ENV PATH="/venv/bin:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
ENV PIP_INSTALL_PATH=/venv/lib/python3.10/site-packages
