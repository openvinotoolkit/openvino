FROM ubuntu:20.04

LABEL version=2020.07.09.1

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
        autoconf \
        automake \
        build-essential \
        cmake \
        curl \
        git \
        libtool \
        ocl-icd-opencl-dev \
        pkg-config \
        unzip \
        wget \
# Python dependencies
        python3 \
        python3-pip \
        python3-dev \
        python3-virtualenv \
        cython3 \
        tox \
# ONNX dependencies
        git-lfs \
        protobuf-compiler \
        libprotobuf-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Build OpenVINO
COPY . /openvino/
WORKDIR /openvino/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_VPU=OFF \
    -DENABLE_GNA=OFF \
    -DENABLE_OPENCV=OFF \
    -DENABLE_CPPLINT=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_BEH_TESTS=OFF \
    -DENABLE_FUNCTIONAL_TESTS=OFF \
    -DENABLE_MKL_DNN=ON \
    -DENABLE_CLDNN=OFF \
    -DENABLE_PROFILING_ITT=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_SPEECH_DEMO=OFF \
    -DENABLE_PYTHON=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DNGRAPH_ONNX_IMPORT_ENABLE=ON \
    -DNGRAPH_INTERPRETER_ENABLE=ON \
    -DNGRAPH_DEBUG_ENABLE=OFF \
    -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=ON \
    -DCMAKE_INSTALL_PREFIX=/openvino/dist
RUN make -j $(nproc) install

# Run tests via tox
WORKDIR /openvino/ngraph/python
ENV NGRAPH_CPP_BUILD_PATH=/openvino/dist
ENV LD_LIBRARY_PATH=/openvino/dist/lib
ENV NGRAPH_ONNX_IMPORT_ENABLE=TRUE
ENV PYTHONPATH=/openvino/bin/intel64/Release/lib/python_api/python3.8:${PYTHONPATH}
RUN git clone --recursive https://github.com/pybind/pybind11.git -b v2.5.0 --depth 1
CMD tox
