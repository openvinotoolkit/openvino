ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN touch /etc/apt/sources.list.d/ubuntu.sources && \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i -E 's|http://(archive|security).ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
    fi && \
    sed -i -E 's|http://(archive|security).ubuntu.com/ubuntu|http://azure.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
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
        lcov \
        pigz \
        clang \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
        libhdf5-dev \
        xvfb \
        libgtk-3-0 \
        libgbm1 \
        libnss3 \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        clinfo \
        libegl-mesa0 \
        libegl1-mesa \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        libgl1-mesa-dri \
        libglapi-mesa \
        libgles2-mesa-dev \
        libglx-mesa0 \
        libxatracker2 \
        mesa-va-drivers \
        mesa-vdpau-drivers \
        mesa-vulkan-drivers \
        va-driver-all && \
    rm -rf /var/lib/apt/lists/*

ADD scripts/install_dependencies/install_openvino_dependencies.sh /install_openvino_dependencies.sh
RUN chmod +x /install_openvino_dependencies.sh && \
    /install_openvino_dependencies.sh -c=core -c=dev -c=gpu -y && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://raw.githubusercontent.com/google/gtest-parallel/master/gtest_parallel.py -O /usr/local/bin/gtest_parallel.py && \
    chmod +x /usr/local/bin/gtest_parallel.py

ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
ENV PIP_INSTALL_PATH=/venv/lib/python3.11/site-packages