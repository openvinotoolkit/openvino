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

RUN apt-get update && \
    apt-get install software-properties-common && \
    apt-get update && \
    apt-get install \
        curl \
        wget \
        git \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Samples
        clang \
        # Python
        python3-full \
        libhdf5-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install openvino dependencies
ADD scripts/install_dependencies/install_openvino_dependencies.sh /install_openvino_dependencies.sh
RUN chmod +x /install_openvino_dependencies.sh && \
    /install_openvino_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# `gh` CLI
RUN mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for the system Python as Python in Ubuntu 24 complains about installing
# packages into the system Python
RUN python3 -m venv venv
ENV PATH="/venv/bin:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN /venv/bin/python3 -m pip install --upgrade pip==${PIP_VERSION}

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
ENV PIP_INSTALL_PATH=/venv/lib/python3/site-packages
