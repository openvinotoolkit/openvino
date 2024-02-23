FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04

USER root

# Set apt retries
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries

RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        # Install git
        git ca-certificates \
        # Install Python
        python3.11-dev \
        # For Java API
        default-jdk \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
COPY install_build_dependencies.sh /install_build_dependencies.sh
RUN /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*
