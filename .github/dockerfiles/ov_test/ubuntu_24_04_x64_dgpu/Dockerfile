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
        git \
        gpg-agent \
        tzdata \
        wget \
        clinfo \
        ca-certificates \
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
    /install_openvino_dependencies.sh -c=core -c=dev -c=gpu -y && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://raw.githubusercontent.com/google/gtest-parallel/master/gtest_parallel.py -O /usr/local/bin/gtest_parallel.py && \
    chmod +x /usr/local/bin/gtest_parallel.py

# Install dgpu drivers
RUN add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update && apt-get install ocl-icd-libopencl1 libze-dev libze1 && \
    mkdir drivers && \
    cd drivers && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.32.7/intel-igc-core-2_2.32.7+21184_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.32.7/intel-igc-opencl-2_2.32.7+21184_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-ocloc-dbgsym_26.14.37833.4-0_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-ocloc_26.14.37833.4-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-opencl-icd-dbgsym_26.14.37833.4-0_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/intel-opencl-icd_26.14.37833.4-0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libigdgmm12_22.9.0_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libze-intel-gpu1-dbgsym_26.14.37833.4-0_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/libze-intel-gpu1_26.14.37833.4-0_amd64.deb && \
    dpkg -i *.deb && cd .. && rm -rf drivers

# Create a virtual environment for the system Python as Python in Ubuntu 24 complains about installing
# packages into the system Python
RUN python3 -m venv venv
ENV PATH="/venv/bin:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN /venv/bin/python3 -m pip install --upgrade pip==${PIP_VERSION}

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
ENV PIP_INSTALL_PATH=/venv/lib/python3/site-packages
