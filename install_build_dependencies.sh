#!/bin/bash

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

params=$1

yes_or_no() {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p -r "Add third-party Nux Dextop repository and install FFmpeg package (y) / Skip this step (N)" yn
        case $yn in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
        esac
    done
}

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
    apt-get install -y \
        file \
        `# build tools` \
        build-essential \
        "${cmake_packages[@]}" \
        "${x86_64_specific_packages[@]}" \
        `# to find dependencies` \
        pkg-config \
        `# to deternime product version via git` \
        git \
        `# to speed-up build` \
        ccache \
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
        `# GPU plugin extensions` \
        libva-dev \
        `# python API` \
        python3-pip \
        python3-venv \
        python3-setuptools \
        libpython3-dev \
        `# spell checking for MO sources` \
        python3-enchant \
        `# samples and tools` \
        libgflags-dev \
        zlib1g-dev
    # git-lfs is not available on debian9
    if apt-cache search --names-only '^git-lfs'| grep -q git-lfs; then
        apt-get install -y git-lfs
    fi
    # for python3-enchant
    if apt-cache search --names-only 'libenchant1c2a'| grep -q libenchant1c2a; then
        apt-get install -y libenchant1c2a
    fi
    # samples
    if apt-cache search --names-only '^nlohmann-json3-dev'| grep -q nlohmann-json3; then
        apt-get install -y nlohmann-json3-dev
    else
        apt-get install -y nlohmann-json-dev
    fi
elif [ -f /etc/redhat-release ] || grep -q "rhel" /etc/os-release ; then
    # RHEL 8 / CentOS 7
    yum update
    yum install -y centos-release-scl epel-release
    yum install -y \
        file \
        `# build tools`
        cmake3 \
        gcc \
        gcc-c++ \
        make \
        `# to determine openvino version via git` \
        git \
        git-lfs \
        `# to speed-up build` \
        ccache \
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
        `# GPU plugin dependency`
        libva-devel
        `# python API` \
        python3-pip \
        python3-devel \
        `# samples and tools` \
        zlib-devel \
        gflags-devel
elif [ -f /etc/os-release ] && grep -q "raspbian" /etc/os-release; then
    # Raspbian
    apt update
    apt-get install -y \
        file \
        `# build tools` \
        build-essential \
        `# to find dependencies` \
        pkg-config \
        `# to deternime product version via git` \
        git \
        `# to speed-up build` \
        ccache \
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
        libpython3-dev \
        `# samples and tools` \
        libgflags-dev \
        zlib1g-dev \
        nlohmann-json-dev
else
    echo "Unknown OS, please install build dependencies manually"
fi

# cmake 3.20.0 or higher is required to build OpenVINO
current_cmake_ver=$(cmake --version | sed -ne 's/[^0-9]*\(\([0-9]\.\)\{0,4\}[0-9][^.]\).*/\1/p')
required_cmake_ver=3.20.0
if [ ! "$(printf '%s\n' "$required_cmake_ver" "$current_cmake_ver" | sort -V | head -n1)" = "$required_cmake_ver" ]; then
    installed_cmake_ver=3.23.2
    arch=$(uname -m)
    wget "https://github.com/Kitware/CMake/releases/download/v${installed_cmake_ver}/cmake-${installed_cmake_ver}-linux-${arch}.sh"
    chmod +x "cmake-${installed_cmake_ver}-linux-${arch}.sh"
    "./cmake-${installed_cmake_ver}-linux-${arch}.sh" --skip-license --prefix=/usr/local
    rm -rf "cmake-${installed_cmake_ver}-linux-${arch}.sh"
fi
