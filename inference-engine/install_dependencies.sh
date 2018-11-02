#!/bin/bash -x
# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

system_ver=`lsb_release -r | cut -d ":" -f2 | sed 's/^[\t]*//g'`

# install dependencies
if [[ -f /etc/lsb-release ]]; then
    # Ubuntu
    sudo -E apt update
    sudo -E apt-get install -y \
            build-essential \
            cmake \
            curl \
            wget \
            libssl-dev \
            ca-certificates \
            git \
            libboost-regex-dev \
            gcc-multilib \
            g++-multilib \
            libgtk2.0-dev \
            pkg-config \
            unzip \
            automake \
            libtool \
            autoconf \
            libcairo2-dev \
            libpango1.0-dev \
            libglib2.0-dev \
            libgtk2.0-dev \
            libswscale-dev \
            libavcodec-dev \
            libavformat-dev \
            libgstreamer1.0-0 \
            gstreamer1.0-plugins-base \
            libusb-1.0-0-dev \
            libopenblas-dev
    if [ $system_ver = "18.04" ]; then
	    sudo -E apt-get install -y libpng-dev
    else
	    sudo -E apt-get install -y libpng12-dev
    fi 
else
    # CentOS 7.x
    sudo -E yum install -y centos-release-scl epel-release
    sudo -E yum install -y \
            wget \
            tar \
            xz \
            p7zip \
            unzip \
            yum-plugin-ovl \
            which \
            libssl-dev \
            ca-certificates \
            git \
            boost-devel \
            libtool \
            gcc \
            gcc-c++ \
            make \
            glibc-static \
            glibc-devel \
            libstdc++-static \
            libstdc++-devel \
            libstdc++ libgcc \
            glibc-static.i686 \
            glibc-devel.i686 \
            libstdc++-static.i686 \
            libstdc++.i686 \
            libgcc.i686 \
            libusbx-devel \
            openblas-devel

    wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz --no-check-certificate
    tar xf cmake-3.12.3.tar.gz
    cd cmake-3.12.3
    ./configure
    make -j16
    sudo -E make install

    # FFmpeg and GStreamer for OpenCV
    sudo -E rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm
    sudo -E yum install -y ffmpeg libusbx-devel gstreamer1 gstreamer1-plugins-base
fi
