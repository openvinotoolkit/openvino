FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries && \
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
        ca-certificates \
        gpg-agent \
        tzdata \
        # Pythons
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        # For Java API
        default-jdk \
        # For building the latest h5py
        libhdf5-dev \
        # Compiler
        gcc-10 \
        g++-10 \
        && \
    rm -rf /var/lib/apt/lists/*

# Setup pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py  --no-cache-dir pip==24.0 && \
    python3.11 get-pip.py --no-cache-dir pip==24.0 && \
    rm -f get-pip.py

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Set gcc-10 as a default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30

# Use Python 3.11 as default instead of Python 3.8
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"
