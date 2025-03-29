#!/bin/bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [ $EUID -ne 0 ]; then
    echo "ERROR: this script must be run as root to install 3rd party packages." >&2
    echo "Please try again with \"sudo -E $0\", or as root." >&2
    exit 1
fi

# install dependencies
if [ -f /etc/lsb-release ] || [ -f /etc/debian_version ] ; then
    # Ubuntu
    host_cpu=$(uname -m)

    x86_64_specific_packages=()
    if [ "$host_cpu" = "x86_64" ]; then
        # to build 32-bit or ARM binaries on 64-bit host
        x86_64_specific_packages+=(gcc-multilib g++-multilib)
    fi

    if ! command -v cmake &> /dev/null; then
        cmake_packages=(cmake)
    fi

    apt update
    apt-get install -y --no-install-recommends \
        `# for python3-pip` \
        ca-certificates \
        file \
        `# build tools` \
        build-essential \
        ninja-build \
        scons \
        ccache \
        "${cmake_packages[@]}" \
        "${x86_64_specific_packages[@]}" \
        `# to find dependencies` \
        pkgconf \
        `# to deternime product version via git` \
        git \
        `# check bash scripts for correctness` \
        shellcheck \
        `# to build and check pip packages` \
        patchelf \
        fdupes \
        `# archive debian changelog file` \
        gzip \
        `# to check debian package correctness` \
        lintian \
        `# openvino main dependencies` \
        libtbb-dev \
        libpugixml-dev \
        `# OpenCL for GPU` \
        ocl-icd-opencl-dev \
        opencl-headers \
        rapidjson-dev \
        `# GPU plugin extensions` \
        libva-dev \
        `# For TF FE saved models` \
        libsnappy-dev \
        `# python API` \
        python3-pip \
        python3-venv \
        python3-setuptools \
        libpython3-dev \
        pybind11-dev \
        libffi-dev \
        `# spell checking for MO sources` \
        python3-enchant \
        `# tools` \
        wget
    # TF lite frontend
    if apt-cache search --names-only '^libflatbuffers-dev'| grep -q libflatbuffers-dev; then
        apt-get install -y --no-install-recommends libflatbuffers-dev
    fi
    # git-lfs is not available on debian9
    if apt-cache search --names-only '^git-lfs'| grep -q git-lfs; then
        apt-get install -y --no-install-recommends git-lfs
    fi
    # for python3-enchant
    if apt-cache search --names-only 'libenchant1c2a'| grep -q libenchant1c2a; then
        apt-get install -y --no-install-recommends libenchant1c2a
    fi
    # samples
    if apt-cache search --names-only '^nlohmann-json3-dev'| grep -q nlohmann-json3; then
        apt-get install -y --no-install-recommends nlohmann-json3-dev
    else
        apt-get install -y --no-install-recommends nlohmann-json-dev
    fi
elif [ -f /etc/redhat-release ] || grep -q "rhel\|tencentos\|opencloudos" /etc/os-release ; then
    yum update
    # RHEL 8 / CentOS 7
    if [ -f /etc/redhat-release ] || grep -q "rhel" /etc/os-release ; then
        yum install -y centos-release-scl
        yum install -y epel-release
        yum install -y \
            `# to build and check pip packages` \
            patchelf \
            `# check bash scripts for correctness` \
            ShellCheck
    else
        yum install -y epol-release
    fi
    yum install -y \
        file \
        `# build tools` \
        cmake3 \
        ccache \
        ninja-build \
        scons \
        gcc \
        gcc-c++ \
        make \
        `# to determine openvino version via git` \
        git \
        fdupes \
        `# to build and check rpm packages` \
        rpm-build \
        rpmlint \
        `# main openvino dependencies` \
        tbb-devel \
        pugixml-devel \
        `# GPU plugin dependency` \
        libva-devel \
        `# For TF FE saved models` \
        snappy-devel \
        `# OpenCL for GPU` \
        ocl-icd-devel \
        opencl-headers \
        `# python API` \
        python3-pip \
        python3-devel
elif [ -f /etc/os-release ] && grep -q "SUSE" /etc/os-release ; then
    zypper refresh
    zypper install -y \
        file \
        `# build tools` \
        patterns-devel-C-C++-devel_C_C++ \
        cmake \
        ccache \
        ninja \
        scons \
        gcc \
        gcc-c++ \
        make \
        `# to determine openvino version via git` \
        git \
        `# to build and check pip packages` \
        patchelf \
        fdupes \
        `# to build and check rpm packages` \
        rpm-build \
        rpmlint \
        `# check bash scripts for correctness` \
        ShellCheck \
        `# main openvino dependencies` \
        tbb-devel \
        pugixml-devel \
        `# GPU plugin dependency` \
        libva-devel \
        `# For TF FE saved models` \
        snappy-devel \
        `# OpenCL for GPU` \
        ocl-icd-devel \
        opencl-cpp-headers \
        opencl-headers \
        `# python API` \
        python39-pip \
        python39-setuptools \
        python39-devel
elif [ -f /etc/os-release ] && grep -q "raspbian" /etc/os-release; then
    # Raspbian
    apt update
    apt-get install -y --no-install-recommends \
        file \
        `# build tools` \
        build-essential \
        ccache \
        ninja-build \
        scons \
        `# to find dependencies` \
        pkg-config \
        `# to determine product version via git` \
        git \
        `# to build and check pip packages` \
        patchelf \
        fdupes \
        `# archive debian changelog file` \
        gzip \
        `# openvino main dependencies` \
        libtbb-dev \
        libpugixml-dev \
        `# python API` \
        python3-pip \
        python3-venv \
        python3-setuptools \
        libpython3-dev
elif [ -f /etc/os-release ] && grep -q "void" /etc/os-release; then
    #Void Linux
    xbps-install -Syu
    xbps-install -y \
        `# for python3-pip` \
        `# ca-certificates (already included)` \
        file \
        `# build tools` \
        base-devel \
        ninja \
        scons \
        ccache \
        cmake \
        `# to find dependencies` \
        pkgconf \
        `# to determine product version via git` \
        git \
        `# to check bash scripts for correctness` \
        shellcheck \
        `# to build and check pip packages` \
        patchelf \
        fdupes \
        `# main openvino dependencies` \
        tbb-devel \
        pugixml-devel \
        `# OpenCL for GPU` \
        ocl-icd-devel \
        OpenCL-Headers \
        OpenCL-CLHPP \
        rapidjson \
        `# GPU plugin dependency` \
        libva-devel \
        `# For TF FE saved models` \
        snappy-devel \
        `# For Python API` \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        python3-devel \
        python3-pybind11 \
        libffi-devel \
        `# Spell checking for MO sources` \
        python3-enchant \
        `# tools` \
        wget \
        git-lfs \
        `# TF Lite Frontend` \
        flatbuffers-devel \
        `# for python3-enchant` \
        enchant2-devel \
        `# samples` \
        json-c++
elif [ -f /etc/os-release ] && grep -q "alpine" /etc/os-release; then
    #Alpine Linux
    apk --no-cache add \
        `# for python3-pip` \
	ca-certificates \
        file \
        `# build tools` \
        build-base \
        ninja-is-really-ninja \
        scons \
        ccache \
        cmake \
        `# to find dependencies` \
        pkgconf \
        `# to determine product version via git` \
        git \
        `# to check bash scripts for correctness` \
        shellcheck \
        `# to build and check pip packages` \
        patchelf \
        fdupes \
        `# main openvino dependencies` \
        onetbb-dev \
        py3-tbb \
        pugixml-dev \
        `# OpenCL for GPU` \
        opencl-dev `#(includes opencl-headers)`\
        rapidjson-dev \
        `# GPU plugin dependency` \
        libva-dev \
        `# For TF FE saved models` \
        snappy-dev \
        `# For Python API` \
        py3-pip `#(includes py3-setuptools)`\
        py3-wheel \
        py3-virtualenv \
        python3-dev \
        py3-pybind11-dev \
        libffi-dev \
        `# Spell checking for MO sources` \
        py3-enchant \
        `# tools` \
        wget \
        git-lfs \
        `# TF Lite Frontend` \
        flatbuffers-dev \
        `# for python3-enchant` \
        enchant2 \
        `# samples` \
        nlohmann-json
else
    echo "Unknown OS, please install build dependencies manually"
fi

# cmake 3.20.0 or higher is required to build OpenVINO

if command -v cmake &> /dev/null; then
    cmake_command=cmake
elif command -v cmake3 &> /dev/null; then
    cmake_command=cmake3
fi

current_cmake_ver=$($cmake_command --version | sed -ne 's/[^0-9]*\(\([0-9]\.\)\{0,4\}[0-9][^.]\).*/\1/p')
required_cmake_ver=3.24.0
if [ ! "$(printf '%s\n' "$required_cmake_ver" "$current_cmake_ver" | sort -V | head -n1)" = "$required_cmake_ver" ]; then
    installed_cmake_ver=3.26.0
    arch=$(uname -m)

    if command -v apt-get &> /dev/null; then
        apt-get install -y --no-install-recommends wget
    elif command -v yum &> /dev/null; then
        yum install -y wget
    elif command -v zypper &> /dev/null; then
        zypper in -y wget
    fi

    cmake_install_bin="cmake-${installed_cmake_ver}-linux-${arch}.sh"
    github_cmake_release="https://github.com/Kitware/CMake/releases/download/v${installed_cmake_ver}/${cmake_install_bin}"
    wget "${github_cmake_release}" -O "${cmake_install_bin}"
    chmod +x "${cmake_install_bin}"
    "./${cmake_install_bin}" --skip-license --prefix=/usr/local
    rm -rf "${cmake_install_bin}"
fi
