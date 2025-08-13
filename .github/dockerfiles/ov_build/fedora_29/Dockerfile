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
    xz

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Build Pythons
# Python 3.9
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tar.xz && \
    tar xvf Python-3.9.21.tar.xz
RUN cd /usr/src/Python-3.9.21 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions --prefix=/usr && \
    make altinstall

# Python 3.10
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tar.xz && \
    tar xvf Python-3.10.16.tar.xz
RUN cd /usr/src/Python-3.10.16 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions --prefix=/usr && \
    make altinstall

# Python 3.11
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz && \
    tar xvf Python-3.11.9.tar.xz
RUN cd /usr/src/Python-3.11.9 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions --prefix=/usr && \
    make altinstall

# Python 3.12
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.12.9/Python-3.12.9.tar.xz && \
    tar xvf Python-3.12.9.tar.xz
RUN cd /usr/src/Python-3.12.9 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions --prefix=/usr && \
    make altinstall

# Python 3.13
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tar.xz && \
    tar xvf Python-3.13.2.tar.xz
RUN cd /usr/src/Python-3.13.2 && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions --prefix=/usr && \
    make altinstall

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

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

# To build git and curl from source
RUN dnf update -y && dnf install -y dh-autoreconf \
    curl-devel \
    expat-devel \
    gettext-devel \
    openssl-devel \
    perl-devel \
    zlib-devel \
    libpsl-devel

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

# `gh` cli - Download and install GitHub CLI
RUN curl -L https://github.com/cli/cli/releases/download/v2.76.2/gh_2.76.2_linux_amd64.tar.gz -o /tmp/gh.tar.gz && \
    tar -xzf /tmp/gh.tar.gz -C /opt && \
    mv /opt/gh_2.76.2_linux_amd64 /opt/gh && \
    rm /tmp/gh.tar.gz

ENV PATH="/opt/gh/bin:$PATH"
