FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04

USER root

ARG OPENVINO_ROOT="../../../.."

# Set apt retries
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries

RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        # Install git
        git ca-certificates && \
        # Install Python
        python3.11-dev \
        python3.11-venv \
        # For Java API
        default-jdk \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
COPY ${OPENVINO_ROOT}/install_build_dependencies.sh /install_build_dependencies.sh
RUN /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*
