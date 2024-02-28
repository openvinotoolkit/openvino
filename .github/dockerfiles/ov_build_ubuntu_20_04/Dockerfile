FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:20.04

USER root

# Set apt retries
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80-retries

# to prevent apt-get from waiting user input
ENV DEBIAN_FRONTEND="noninteractive"

# to prevent tzdata from waiting user input
ENV TZ="Europe/London"

RUN apt-get update && \
    apt-get install --assume-yes --no-install-recommends software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --assume-yes --no-install-recommends \
        curl \
        # Install git
        git \
        ca-certificates \
        gpg-agent \
        tzdata \
        # For Java API
        default-jdk \
        # Pythons
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        && \
    rm -rf /var/lib/apt/lists/*

# Setup pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py  --no-cache-dir pip==24.0 && \
    python3.11 get-pip.py --no-cache-dir pip==24.0 && \
    rm -f get-pip.py

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Use Python 3.11 as default instead of Python 3.8
# Using venv here 'cause other methods to switch the default Python on Ubuntu 20 break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$PATH"
