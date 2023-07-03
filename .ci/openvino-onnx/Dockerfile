FROM ubuntu:23.04

LABEL version=2021.03.30.1

# Build configuration arguments
ARG BUILD_TYPE=Release

ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

ENV CI=true
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

# Install base dependencies
RUN apt-get update && apt-get install -y locales && apt-get clean autoclean && apt-get autoremove -y

# Set the locale to en_US.UTF-8
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get -y --no-install-recommends install \
# OpenVINO dependencies
        build-essential \
        ninja-build \
        cmake \
        curl \
        git \
        unzip \
        libtbb-dev \
        libpugixml-dev \
        wget \
# Python dependencies
        python3 \
        python3-pip \
        python3-dev \
        pybind11-dev \
        python3-virtualenv \
        cython3 \
        tox && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Build OpenVINO
COPY . /openvino/
WORKDIR /openvino/build
RUN cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_INTEL_GNA=OFF \
    -DENABLE_INTEL_GPU=OFF \
    -DENABLE_HETERO=OFF \
    -DENABLE_MULTI=OFF \
    -DENABLE_AUTO_BATCH=OFF \
    -DENABLE_GAPI_PREPROCESSING=OFF \
    -DENABLE_CPPLINT=OFF \
    -DENABLE_NCC_STYLE=OFF \
    -DENABLE_PROFILING_ITT=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_OV_PADDLE_FRONTEND=OFF \
    -DENABLE_OV_PYTORCH_FRONTEND=ON \
    -DENABLE_OV_TF_FRONTEND=OFF \
    -DENABLE_OPENVINO_DEBUG=OFF \
    -DCMAKE_INSTALL_PREFIX=/openvino/dist
RUN ninja install

# Run tests via tox
WORKDIR /openvino/src/bindings/python
ENV OpenVINO_DIR=/openvino/dist/runtime/cmake
ENV LD_LIBRARY_PATH=/openvino/dist/runtime/lib/intel64:/openvino/dist/runtime/3rdparty/tbb/lib
ENV PYTHONPATH=/openvino/bin/intel64/${BUILD_TYPE}/python:/openvino/tools/mo:${PYTHONPATH}
CMD tox
