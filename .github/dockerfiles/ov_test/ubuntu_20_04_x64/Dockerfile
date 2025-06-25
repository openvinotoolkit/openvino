ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:20.04

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
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install \
        curl \
        git \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Samples
        clang \
        # python prerequisites
        libcairo2-dev \ 
        # Python
        python3.9-dev \
        python3.9-venv \
        python3.9-distutils \
        libhdf5-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install openvino dependencies
ADD scripts/install_dependencies/install_openvino_dependencies.sh /install_openvino_dependencies.sh
RUN chmod +x /install_openvino_dependencies.sh && \
    /install_openvino_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip-3-8.py && \
    python3 get-pip-3-8.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.9 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Use Python 3.9 as default instead of Python 3.8
RUN ln -sfv $(which python3.9) /usr/local/bin/python && \
    ln -sfv $(which python3.9) /usr/local/bin/python3
RUN ln -sfv $(which pip3.9) /usr/local/bin/pip && \
    ln -sfv $(which pip3.9) /usr/local/bin/pip3
RUN pip install pycairo

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}