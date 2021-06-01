#!/bin/bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

#===================================================================================================
# Option parsing

all_comp=(opencv_req opencv_opt python dev myriad dlstreamer installer cl_compiler)
os=${os:-auto}

# public options
interactive=yes
dry=
extra=
print=
comp=()

# private options
keepcache=
selftest=

while :; do
    case $1 in
        -h|-\?|--help)
            echo "Options:"
            echo "  -y          non-interactive run (off)"
            echo "  -n          dry-run, assume no (off)"
            echo "  -c=<name>   install component <name>, can be repeated (${all_comp[*]})"
            echo "  -e          add extra repositories (CentOS 7) (off)"
            echo "  -p          print package list and exit (off)"
            exit
            ;;
        -y) interactive= ;;
        -n) dry=yes ;;
        -c=?*) comp+=("${1#*=}") ;;
        -e) extra=yes ;;
        -p) print=yes ;;
        --selftest) selftest=yes ;;
        --keepcache) keepcache=yes ;;
        *) break ;;
    esac
    shift
done

# No components selected - install all
if [ ${#comp[@]} -eq 0 ]; then
    comp=(${all_comp[@]})
fi

#===================================================================================================
#  Selftest

if [ -n "$selftest" ] ; then
    for image in centos:7 ubuntu:18.04 ubuntu:20.04 ; do
        for opt in  "-h" "-p" "-e -p" "-n" "-n -e" "-y" "-y -e" ; do
            echo "||"
            echo "|| Test $image / '$opt'"
            echo "||"
            SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"
            docker run -it --rm \
                --volume ${SCRIPT_DIR}:/scripts:ro,Z  \
                --volume yum-cache:/var/cache/yum \
                --volume apt-cache:/var/cache/apt/archives \
                -e DEBIAN_FRONTEND=noninteractive \
                $image \
                bash /scripts/${0##*/} $opt --keepcache
            echo "||"
            echo "|| Completed: $image / '$opt'"
            echo "||"
        done
    done
    echo "Self test finished, to remove temporary docker volumes run:
        'docker volume rm yum-cache apt-cache'"
    exit 0
fi

#===================================================================================================
# OS detection

if [ "$os" == "auto" ] ; then
    os=$( . /etc/os-release ; echo "${ID}${VERSION_ID}" )
    if [[ "$os" =~ "rhel8".* ]] ; then
      os="rhel8"
    fi
    case $os in
        centos7|rhel8|ubuntu18.04|ubuntu20.04) [ -z "$print" ] && echo "Detected OS: ${os}" ;;
        *) echo "Unsupported OS: ${os:-detection failed}" >&2 ; exit 1 ;;
    esac
fi

#===================================================================================================
# Collect packages

extra_repos=()

if [ "$os" == "ubuntu18.04" ] ; then

    pkgs_opencv_req=(libgtk-3-0 libgl1)
    pkgs_python=(python3 python3-dev python3-venv python3-setuptools python3-pip)
    pkgs_dev=(cmake g++ gcc libc6-dev make curl sudo)
    pkgs_myriad=(libusb-1.0-0)
    pkgs_installer=(cpio)
    pkgs_cl_compiler=(libtinfo5)
    pkgs_opencv_opt=(
        gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-ugly
        gstreamer1.0-tools
        libavcodec57
        libavformat57
        libavresample3
        libavutil55
        libgstreamer1.0-0
        libswscale4
    )
    pkgs_dlstreamer=(
        ffmpeg
        flex
        gstreamer1.0-alsa
        gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-ugly
        gstreamer1.0-vaapi
        gstreamer1.0-tools
        libfaac0
        libfluidsynth1
        libgl-dev
        libglib2.0-dev
        libgstreamer1.0-0
        libnettle6
        libtag-extras1
        python3-gi
        vainfo
    )

elif [ "$os" == "ubuntu20.04" ] ; then

    pkgs_opencv_req=(libgtk-3-0 libgl1)
    pkgs_python=(python3 python3-dev python3-venv python3-setuptools python3-pip)
    pkgs_dev=(cmake g++ gcc libc6-dev make curl sudo)
    pkgs_myriad=(libusb-1.0-0)
    pkgs_installer=(cpio)
    pkgs_cl_compiler=(libtinfo5)
    pkgs_opencv_opt=(
        gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-ugly
        gstreamer1.0-tools
        libavcodec58
        libavformat58
        libavresample4
        libavutil56
        libgstreamer1.0-0
        libswscale5
    )
    pkgs_dlstreamer=(
        ffmpeg
        flex
        gstreamer1.0-alsa
        gstreamer1.0-libav
        gstreamer1.0-plugins-bad
        gstreamer1.0-plugins-base
        gstreamer1.0-plugins-good
        gstreamer1.0-plugins-ugly
        gstreamer1.0-vaapi
        gstreamer1.0-tools
        gstreamer1.0-x
        libfaac0
        libfluidsynth2
        libgl-dev
        libglib2.0-dev
        libgstreamer-plugins-base1.0-dev
        libgstreamer1.0-0
        libgstrtspserver-1.0-dev
        libnettle7
        libopenexr24
        libtag-extras1
        python3-gi
        python3-gi-cairo
        python3-gst-1.0
        vainfo
    )

elif [ "$os" == "rhel8" ] ; then

    pkgs_opencv_req=(gtk3)
    pkgs_python=(python3 python3-devel python3-setuptools python3-pip)
    pkgs_dev=(gcc gcc-c++ make glibc libstdc++ libgcc cmake curl sudo)
    pkgs_myriad=()
    pkgs_installer=()
    pkgs_opencv_opt=(
        gstreamer1
        gstreamer1-plugins-bad-free
        gstreamer1-plugins-good
        gstreamer1-plugins-ugly-free
    )
    pkgs_dlstreamer=()
    extra_repos+=(https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm)

elif [ "$os" == "centos7" ] ; then

    # find -name *.so -exec objdump -p {} \; | grep NEEDED | sort -u | cut -c 23- | xargs -t -n1 yum -q whatprovides

    pkgs_opencv_req=(gtk2)
    pkgs_python=(python3 python3-devel python3-setuptools python3-pip)
    pkgs_dev=(gcc gcc-c++ make glibc libstdc++ libgcc cmake curl sudo)
    pkgs_myriad=(libusbx)
    pkgs_installer=()
    pkgs_cl_compiler=()
    pkgs_opencv_opt=(
        gstreamer1
        gstreamer1-plugins-bad-free
        gstreamer1-plugins-good
        gstreamer1-plugins-ugly-free
    )
    pkgs_dlstreamer=(
        OpenEXR-libs
        alsa-lib
        boost-regex
        bzip2-libs
        cairo
        cdparanoia-libs
        flac-libs
        flite
        gdk-pixbuf2
        glib2
        glibc
        gmp
        gsm
        gstreamer1
        gstreamer1-plugins-bad-free
        gstreamer1-plugins-base
        ilmbase
        libX11
        libXdamage
        libXext
        libXfixes
        libXrandr
        libXrender
        libXv
        libdrm
        libdv
        libgcc
        libglvnd-glx
        libjpeg-turbo
        libogg
        libpng
        librdkafka
        librsvg2
        libsndfile
        libsoup
        libstdc++
        libtheora
        libuuid
        libv4l
        libvisual
        libvorbis
        libxml2
        mpg123-libs
        neon
        nettle
        openjpeg2
        openssl-libs
        opus
        orc
        pango
        pulseaudio-libs
        sbc
        soundtouch
        speex
        wavpack
        xz-libs
        zlib
        python36-gi
        python36-gobject
        python36-gobject-devel
    )

    if [ -n "$extra" ] ; then
        # 1 RPMFusion
        extra_repos+=(https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm)
        pkgs_opencv_opt+=(ffmpeg-libs)
        pkgs_dlstreamer+=(
            libde265
            libmms
            librtmp
            opencore-amr
            vo-amrwbenc
        )
        # 2 EPEL
        extra_repos+=(https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm)
        pkgs_dlstreamer+=(
            fluidsynth-libs
            game-music-emu
            libass
            libbs2b
            libchromaprint
            libmodplug
            openal-soft
            paho-c
            spandsp
            zbar
            zvbi
        )
        # 3 ForensicsTools
        extra_repos+=(https://forensics.cert.org/cert-forensics-tools-release-el7.rpm)
        pkgs_dlstreamer+=(
            faac
            fdk-aac
        )
    fi

else
    echo "Internal script error: invalid OS after check (package selection)" >&2
    exit 3
fi

#===================================================================================================
# Gather packages and print list

pkgs=()
for comp in ${comp[@]} ; do
    var=pkgs_${comp}[@]
    pkgs+=(${!var})
done

if [ ${#pkgs[@]} -eq 0 ]; then
    if  [ -n "$print" ] ; then
        echo "No packages to install" >&2
        exit 1
    else
        echo "No packages to install"
        exit 0
    fi
fi

if  [ -n "$print" ] ; then
    echo "${pkgs[*]}"
    exit 0
fi

#===================================================================================================
# Actual installation

if [ $EUID -ne 0 ]; then
    echo "ERROR: this script must be run as root to install 3rd party packages." >&2
    echo "Please try again with \"sudo -E $0\", or as root." >&2
    exit 1
fi

iopt=

if [ "$os" == "ubuntu18.04" ] || [ "$os" == "ubuntu20.04" ] ; then

    [ -z "$interactive" ] && iopt="-y"
    [ -n "$dry" ] && iopt="--dry-run"
    [ -n "$keepcache" ] && rm -f /etc/apt/apt.conf.d/docker-clean

    apt-get update && apt-get install --no-install-recommends $iopt ${pkgs[@]}

elif [ "$os" == "centos7" ] || [ "$os" == "rhel8" ] ; then

    [ -z "$interactive" ] && iopt="--assumeyes"
    [ -n "$dry" ] && iopt="--downloadonly"
    [ -n "$keepcache" ] && iopt="$iopt --setopt=keepcache=1"
    [ ${#extra_repos[@]} -ne 0 ] && yum localinstall $iopt --nogpgcheck ${extra_repos[@]}

    yum install $iopt ${pkgs[@]}

else
    echo "Internal script error: invalid OS after check (package installation)" >&2
    exit 3
fi

exit 0
