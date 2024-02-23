FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04

USER root

# Set apt retries
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries

RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        curl \
        # Install git
        git \
        ca-certificates \
        # Install Python
        python3-distutils \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        # For Java API
        default-jdk \
        && \
    rm -rf /var/lib/apt/lists/*

# Link python3 to python3.11
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3

# Setup pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py  --no-cache-dir pip==24.0 && \
    python3.11 get-pip.py --no-cache-dir pip==24.0 && \
    rm -f get-pip.py

# Install build dependencies
COPY install_build_dependencies.sh /install_build_dependencies.sh
RUN /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*
