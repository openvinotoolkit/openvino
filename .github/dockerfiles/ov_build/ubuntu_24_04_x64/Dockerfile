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
        # Pythons
        python3.9-dev \
        python3.9-venv \
        python3.9-distutils \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        python3.12-dev \
        python3.12-venv \
        python3.13-dev \
        python3.13-venv \
        python3-dev \
        python3-venv \
        python3-pip \
        python3-full \
        # For Java API
        default-jdk \
        # For clang-tidy validation
        clang-format-18 \
        clang-tidy-18 \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install sccache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Create a virtual environment for the system Python as Python in Ubuntu 24 complains about installing
# packages into the system Python
RUN python3 -m venv venv
ENV PATH="/venv/bin:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN /venv/bin/python3 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.9 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.10 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.11 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.12 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.13 -m pip install --upgrade pip==${PIP_VERSION}

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# `gh` cli - Download and install GitHub CLI
RUN curl -L https://github.com/cli/cli/releases/download/v2.76.2/gh_2.76.2_linux_amd64.tar.gz -o /tmp/gh.tar.gz && \
    tar -xzf /tmp/gh.tar.gz -C /opt && \
    mv /opt/gh_2.76.2_linux_amd64 /opt/gh && \
    rm /tmp/gh.tar.gz

ENV PATH="/opt/gh/bin:$PATH"
