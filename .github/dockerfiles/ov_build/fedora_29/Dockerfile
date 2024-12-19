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
    /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Setup Python
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tar.xz && \
    tar xvf Python-3.9.9.tar.xz
RUN cd /usr/src/Python-3.9.9 && \
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
    rm -f get-pip.py

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

