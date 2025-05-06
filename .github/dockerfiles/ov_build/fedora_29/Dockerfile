ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:29

USER root

# dnf configuration
RUN echo "timeout=60" >> /etc/dnf/dnf.conf && \
    echo "retries=10" >> /etc/dnf/dnf.conf

RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3 \
    # To build Python from source
    openssl-devel \
    sqlite-devel \
    bzip2-devel \
    libffi-devel \
    zlib-devel \
    wget \
    make \
    tar \
    gcc \
    gcc-c++ \
    # parallel gzip
    pigz \
    # To build git from source
    xz \
    dh-autoreconf \
    curl-devel \
    expat-devel \
    gettext-devel \
    openssl-devel \
    perl-devel \
    zlib-devel \
    libpsl-devel

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Build and install newer curl
RUN wget https://github.com/curl/curl/releases/download/curl-8_13_0/curl-8.13.0.tar.gz && \
    tar -zxf curl-8.13.0.tar.gz
RUN cd curl-8.13.0 && \
    ./configure --prefix=/usr --with-openssl && \
    make install

# Build and install newer git
RUN wget -O git-2.49.0.tar.gz https://github.com/git/git/archive/refs/tags/v2.49.0.tar.gz && \
    tar -zxf git-2.49.0.tar.gz
RUN cd git-2.49.0 && \
    make configure && ./configure --prefix=/usr && make install

# Use Python 3.9 as default
RUN python3.9 -m venv venv
ENV PATH="/venv/bin:$PATH"
RUN alternatives --install /usr/bin/python python /usr/bin/python3.9 10

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.10 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.12 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.13 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

