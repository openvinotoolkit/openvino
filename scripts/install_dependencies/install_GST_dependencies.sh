#!/bin/bash

# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [ $EUID -ne 0 ]; then
    echo "ERROR: this script must be run as root to install 3rd party packages." >&2
    echo "Please try again with \"sudo -E $0\", or as root." >&2
    exit 1
fi

params=$@

yes_or_no() {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p "Add third-party repositories and install GStreamer Plugins (y/n): " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) return  1 ;;
        esac
    done
}

echo
echo "This script installs the following GStreamer 3rd-party dependencies:"
echo "  1. build dependencies for GStreamer plugin bad"
echo "  2. build dependencies for GStreamer plugin ugly"
echo "  3. build dependencies for GStreamer plugin vaapi"
echo

if [ -f /etc/lsb-release ]; then
    # Ubuntu
    PKGS=(
        libbluetooth-dev
        libusb-1.0.0-dev
        libass-dev
        libbs2b-dev
        libchromaprint-dev
        liblcms2-dev
        libssh2-1-dev
        libdc1394-22-dev
        libdirectfb-dev
        libssh-dev
        libdca-dev
        libfaac-dev
        libfaad-dev
        libfdk-aac-dev
        flite1-dev
        libfluidsynth-dev
        libgme-dev
        libgsm1-dev
        nettle-dev
        libkate-dev
        liblrdf0-dev
        libde265-dev
        libmjpegtools-dev
        libmms-dev
        libmodplug-dev
        libmpcdec-dev
        libneon27-dev
        libofa0-dev
        libopenal-dev
        libopenexr-dev
        libopenjp2-7-dev
        libopenmpt-dev
        libopenni2-dev
        libdvdnav-dev
        librtmp-dev
        librsvg2-dev
        libsbc-dev
        libsndfile1-dev
        libsoundtouch-dev
        libspandsp-dev
        libsrtp2-dev
        libzvbi-dev
        libvo-aacenc-dev
        libvo-amrwbenc-dev
        libwebrtc-audio-processing-dev
        libwebp-dev
        libwildmidi-dev
        libzbar-dev
        libnice-dev
        libx265-dev
        libxkbcommon-dev
        libx264-dev
        libmpeg2-4-dev
        libdvdread-dev
        libcdio-dev
        libopencore-amrnb-dev
        libopencore-amrwb-dev
        liba52-0.7.4-dev
        libsidplay1-dev
        libva-dev
        libxrandr-dev
        libudev-dev
        python-gi-dev \
        python3-dev
    )
    apt update
    apt install -y ${PKGS[@]}
else
    # CentOS
    PKGS=(
        bluez-libs-devel
        libusb-devel
        libass-devel
        libbs2b-devel
        libchromaprint-devel
        lcms2-devel
        libssh2-devel
        libdc1394-devel
        libXext-devel
        libssh-devel
        libdca-devel
        faac-devel
        faad2-devel
        fdk-aac-devel
        flite-devel
        fluidsynth-devel
        game-music-emu-devel
        gsm-devel
        nettle-devel
        kate-devel
        liblrdf-devel
        libde265-devel
        mjpegtools-devel
        libmms-devel
        libmodplug-devel
        libmpcdec-devel
        neon-devel
        libofa-devel
        openal-soft-devel
        OpenEXR-devel
        openjpeg2-devel
        openni-devel
        libdvdnav-devel
        librtmp-devel
        librsvg2-devel
        sbc-devel
        libsndfile-devel
        soundtouch-devel
        spandsp-devel
        libsrtp-devel
        zvbi-devel
        vo-amrwbenc-devel
        webrtc-audio-processing-devel
        wildmidi-devel
        zbar-devel
        libnice-devel
        x265-devel
        libxkbcommon-devel
        x264-devel
        libmpeg2-devel
        libcdio-devel
        opencore-amr-devel
        libva-devel
        python36-gobject-devel
        python3-devel
    )
    if yes_or_no; then
        rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
        yum install -y epel-release
        rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
        yum install -y ${PKGS[@]}
    else
        echo
        echo "Plugins cannot be installed without adding repositories:"
        echo "     PM-GPG-KEY-nux, epel-release, nux-dextop-release-0-5."
        echo
    fi
    exit
fi
