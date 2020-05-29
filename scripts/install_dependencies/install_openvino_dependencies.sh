#!/bin/bash

# Copyright (c) 2018 Intel Corporation
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

yes_or_no_ffmpeg() {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p "Add third-party RPM Fusion repository and install FFmpeg package (y/n): " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) return  1 ;;
        esac
    done
}

yes_or_no_gst_bad_ugly() {
    if [ "$params" == "-y" ]; then
        return 0
    fi

    while true; do
        read -p "Add third-party RPM Epel, Nux, Fusion, Forensics repositories and install dependencies for GStreamer Bad & Ugly Plugins (y/n): " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) return  1 ;;
        esac
    done
}

if [ -f /etc/lsb-release ]; then
    # Ubuntu
    echo
    echo "This script installs the following OpenVINO 3rd-party dependencies:"
    echo "  1. GTK+, FFmpeg and GStreamer libraries used by OpenCV"
    echo "  2. libusb library required for Myriad plugin for Inference Engine"
    echo "  3. build dependencies for OpenVINO samples"
    echo "  4. build dependencies for GStreamer Plugins"
    echo
    PKGS=(
        cpio
        build-essential
        cmake
        libusb-1.0-0-dev
        libdrm-dev
        libgstreamer1.0-0
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad
        ffmpeg
    )
    system_ver=$(cat /etc/lsb-release | grep -i "DISTRIB_RELEASE" | cut -d "=" -f2)
    if [ "$system_ver" = "16.04" ]; then
        PKGS+=( libgtk2.0-0 )
    else
        PKGS+=( libgtk-3-0
                libglib2.0-0
                flex
                bison
                libgmp10
                libgsl23
                gobject-introspection
                libcap2
                libcap2-bin
                gettext
                libgirepository-1.0-1
                libx11-6
                iso-codes
                libgl1-mesa-dri
                libgles2
                libgl-dev
                gudev-1.0
                libtheora0
                libcdparanoia0
                libpango-1.0-0
                libgbm1
                libasound2
                libjpeg8
                libvisual-0.4-0
                libxv1
                libopus0
                libgraphene-1.0-0
                libvorbis0a
                libbz2-1.0
                libv4l-0
                libaa1
                libflac8
                libgdk-pixbuf2.0-0
                libmp3lame0
                libcaca0
                libdv4
                libmpg123-0
                libraw1394-11
                libavc1394-0
                libiec61883-0
                libpulse0
                libsoup2.4-1
                libspeex1
                libtag-extras1
                libtwolame0
                libwavpack1
                libbluetooth3
                libusb-1.0.0-dev
                libass9
                libbs2b0
                libchromaprint1
                liblcms2-2
                libssh2-1
                libdc1394-22
                libdirectfb-1.7-7
                libssh-4
                libdca0
                libfaac0
                libfdk-aac1
                libflite1
                libfluidsynth1
                libgme0
                libgsm1
                libnettle6
                libkate1
                liblrdf0
                libde265-0
                libmjpegtools-dev
                libmms0
                libmodplug1
                libmpcdec6
                libneon27
                libopenal1
                libopenexr22
                libopenjp2-7
                libopenmpt0
                libopenni2-0
                libdvdnav4
                librtmp1
                librsvg2-2
                libsbc1
                libsndfile1
                libsoundtouch1
                libspandsp2
                libsrtp2-1
                libzvbi0
                libvo-aacenc0
                libvo-amrwbenc0
                libwebrtc-audio-processing1
                libwebp6
                libwildmidi2
                libzbar0
                libnice10
                libxkbcommon0
                libmpeg2-4
                libopencore-amrnb0
                libopencore-amrwb0
                liba52-0.7.4
                libva2
                libxrandr2
                libudev1
                python3.6
                libpython3.6
                python3-gi
        )
    fi
    apt update
    apt install -y ${PKGS[@]}
else
    # CentOS
    echo
    echo "This script installs the following OpenVINO 3rd-party dependencies:"
    echo "  1. GTK+ and GStreamer libraries used by OpenCV"
    echo "  2. libusb library required for Myriad plugin for Inference Engine"
    echo "  3. Python 3.6 for Model Optimizer"
    echo "  4. gcc 4.8.5 and other build dependencies for OpenVINO samples"
    echo "  5. build dependencies for GStreamer Plugins"
    echo
    PKGS=(
        libusbx-devel
        gtk2
        gstreamer1
        gstreamer1-plugins-good
        gstreamer1-plugins-bad-free
        gcc
        gcc-c++
        make
        glibc-static
        glibc
        libstdc++-static
        libstdc++
        libstdc++
        libgcc
        cmake
        python36
        python36-pip
        glib2-devel
        flex
        bison
        gmp
        gsl
        gobject-introspection
        libcap
        libcap
        gettext
        libXrandr
        libX11
        iso-codes
        mesa-libEGL
        mesa-libGLES
        mesa-libGL
        libgudev1
        libtheora
        cdparanoia
        pango
        mesa-libgbm
        alsa-lib
        libjpeg-turbo
        libvisual
        libXv
        opus
        libvorbis
        patch
        bzip2
        libv4l
        flac
        gdk-pixbuf2
        libdv
        mpg123
        libraw1394
        libavc1394
        libiec61883
        pulseaudio-libs
        libsoup
        speex
        wavpack
        boost-regex-1.53.0
    )
    yum install -y ${PKGS[@]}

    # Thirdparty repositories for installing GStreamer Bad & Ugly Plugins dependencies.
    if yes_or_no_gst_bad_ugly; then
        GST_BAD_UGLY_PKGS=(
            bluez-libs
            libusb
            libass
            libbs2b
            libchromaprint
            lcms2
            libssh2
            libdc1394
            libXext
            libssh
            libdca
            faac
            fdk-aac
            flite
            fluidsynth
            game-music-emu
            gsm
            nettle
            kate
            liblrdf
            libde265
            mjpegtools
            libmms
            libmodplug
            libmpcdec
            neon
            openal-soft
            OpenEXR
            openjpeg2
            openni
            libdvdnav
            librtmp
            librsvg2
            sbc
            libsndfile
            soundtouch
            spandsp
            libsrtp
            zvbi
            vo-amrwbenc
            webrtc-audio-processing
            wildmidi
            zbar
            libnice
            libxkbcommon
            libmpeg2
            libcdio
            opencore-amr
            libva
            python36-gobject
            python3-devel
        )
        yum install -y epel-release
        rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
        RPMFUSION_IS_INSTALLED=1
        yum install -y https://forensics.cert.org/cert-forensics-tools-release-el7.rpm
        yum install -y ${GST_BAD_UGLY_PKGS[@]}
    else
        echo "Dependencies for GStreamer Ugly & Bad plugins installation skipped."
        echo
    fi

    echo
    echo "Intel(R) Distribution of OpenVINO(TM) toolkit can use FFmpeg for processing video streams with OpenCV. Please select your preferred method for installing FFmpeg:"
    echo
    echo "Option 1: Allow installer script to add a third party repository, RPM Fusion (https://rpmfusion.org/), which contains FFmpeg. FFmpeg rpm package will be installed from this repository. "
    echo "WARNING: This repository is NOT PROVIDED OR SUPPORTED by Intel or CentOS. Neither Intel nor CentOS has control over this repository. Terms governing your use of FFmpeg can be found here: https://www.ffmpeg.org/legal.html "
    echo "Once added, this repository will be enabled on your operating system and can thus receive updates to all packages installed from it. "
    echo
    echo "Consider the following ways to prevent unintended 'updates' from this third party repository from over-writing some core part of CentOS:"
    echo "a) Only enable these archives from time to time, and generally leave them disabled. See: man yum"
    echo "b) Use the exclude= and includepkgs= options on a per sub-archive basis, in the matching .conf file found in /etc/yum.repos.d/ See: man yum.conf"
    echo "c) The yum Priorities plug-in can prevent a 3rd party repository from replacing base packages, or prevent base/updates from replacing a 3rd party package."
    echo
    echo "Option 2: Skip FFmpeg installation."
    echo

    if yes_or_no_ffmpeg; then
        if [[ -z $RPMFUSION_IS_INSTALLED ]]; then
            yum install -y epel-release
            rpm -Uvh https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
        fi
        yum install -y ffmpeg
    else
        echo "FFmpeg installation skipped. You may build FFmpeg from sources as described here: https://trac.ffmpeg.org/wiki/CompilationGuide/Centos"
        echo
    fi
    exit
fi
