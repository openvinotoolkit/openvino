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
    if [ "$host_cpu" = "x86_64" ]; then
        x86_64_specific_packages=(gcc-multilib g++-multilib)
    else
        x86_64_specific_packages=()
    fi

    if ! command -v cmake &> /dev/null; then
        cmake_packages=(cmake)
    fi

    apt update
    apt-get install -y \
        build-essential \
        "${cmake_packages[@]}" \
        ccache \
        curl \
        wget \
        libssl-dev \
        ca-certificates \
        git \
        "${x86_64_specific_packages[@]}" \
        libgtk2.0-dev \
        unzip \
        shellcheck \
        patchelf \
        fdupes \
        lintian \
        file \
        gzip \
        `# openvino` \
        libtbb-dev \
        libpugixml-dev \
        `# gpu plugin extensions` \
        libva-dev \
        `# python` \
        python3-pip \
        python3-venv \
        python3-enchant \
        python3-setuptools \
        libpython3-dev \
        `# samples` \
        pkg-config \
        libgflags-dev \
        zlib1g-dev \
        `# hddl` \
        libudev1 \
        libusb-1.0-0 \
        `# myriad` \
        libusb-1.0-0-dev \
        `# cl_compiler` \
        libtinfo5
    # git-lfs is not available on debian9
    if apt-cache search --names-only '^git-lfs'| grep -q git-lfs; then
        apt-get install -y git-lfs
    fi
    # hddl
    if apt-cache search --names-only '^libboost-filesystem1.65.1' | grep -q libboost-filesystem1.65.1 ; then
        # ubuntu 18.04
        apt-get install -y \
            libjson-c3 \
            libboost-filesystem1.65.1 \
            libboost-program-options1.65.1 \
            libboost-system1.65.1
    elif apt-cache search --names-only '^libjson-c4'| grep -q libjson-c4; then
        # ubuntu 20.04
        apt-get install -y \
            libjson-c4 \
            libboost-filesystem1.71.0 \
            libboost-program-options1.71.0
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
        wget \
        curl \
        cmake3 \
        tar \
        xz \
        p7zip \
        ccache \
        openssl-devel \
        rpm-build \
        rpmlint \
        ShellCheck \
        unzip \
        which \
        ca-certificates \
        git \
        git-lfs \
        boost-devel \
        python3-pip \
        python3-devel \
        libtool \
        tbb-devel \
        pugixml-devel \
        gcc \
        gcc-c++ \
        make \
        patchelf \
        fdupes \
        libusbx-devel \
        file \
        zlib-devel \
        gflags-devel \
        libva-devel
elif [ -f /etc/os-release ] && grep -q "raspbian" /etc/os-release; then
    # Raspbian
    apt update
    apt-get install -y \
        build-essential \
        wget \
        libssl-dev \
        ca-certificates \
        git \
        pkg-config \
        libgflags-dev \
        zlib1g-dev \
        nlohmann-json-dev \
        unzip \
        libusb-1.0-0-dev
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
