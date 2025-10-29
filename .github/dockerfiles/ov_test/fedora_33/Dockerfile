ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:33

USER root

# dnf configuration
RUN echo "timeout=60" >> /etc/dnf/dnf.conf && \
    echo "retries=10" >> /etc/dnf/dnf.conf

RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3 \
    findutils \
    ocl-icd \
    ocl-icd-devel \
    # parallel gzip
    pigz \
    xz \
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
    gcc-c++

# Build Python 3.10 from source
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tar.xz && \
    tar xvf Python-3.10.16.tar.xz && \
    cd Python-3.10.16 && \
    ./configure --enable-optimizations --prefix=/usr && \
    make altinstall && \
    cd /usr/src && \
    rm -rf Python-3.10.16*

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
