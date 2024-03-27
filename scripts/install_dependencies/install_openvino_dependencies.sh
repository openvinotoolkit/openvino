#!/bin/bash

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

#===================================================================================================
# Option parsing

all_comp=(core dev gpu python)
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
            echo "  -e          add extra repositories (RHEL 7, 8, 9) (off)"
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

# No components selected - install default
if [ ${#comp[@]} -eq 0 ]; then
    comp=("${all_comp[@]}")
fi

#===================================================================================================
#  Selftest

if [ -n "$selftest" ] ; then
    for image in centos:7 centos:8 rhel:8 rhel:9.1 \
                 almalinux:8.7 amazonlinux:2 \
                 fedora:34 fedora:35 fedora:36 fedora:37 fedora:38 \
                 opensuse/leap:15.3 \
                 raspbian:9 debian:9 ubuntu:18.04 \
                 raspbian:10 debian:10 ubuntu:20.04 ubuntu:20.10 ubuntu:21.04 \
                 raspbian:11 debian:11 ubuntu:21.10 ubuntu:22.04 \
                 raspbian:12 debian:12 ubuntu:22.10 ubuntu:23.04 ; do
        for opt in  "-h" "-p" "-e -p" "-n" "-n -e" "-y" "-y -e" ; do
            echo "||"
            echo "|| Test $image / '$opt'"
            echo "||"
            SCRIPT_DIR="$( cd "$( dirname "$(realpath "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"
            docker run -it --rm \
                --volume "${SCRIPT_DIR}":/scripts:ro,Z  \
                --volume yum-cache:/var/cache/yum \
                --volume apt-cache:/var/cache/apt/archives \
                -e DEBIAN_FRONTEND=noninteractive \
                $image \
                bash "/scripts/${0##*/}" "$opt" --keepcache
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
    # shellcheck source=/dev/null
    os=$( . /etc/os-release ; echo "${ID}${VERSION_ID}" )
    if [[ "$os" =~ "rhel8".* ]] ; then
      os="rhel8"
    fi
    case $os in
        centos7|centos8|centos9|\
        rhel8|rhel9.1|\
        anolis8.6|anolis8.8|\
        openEuler20.03|openEuler22.03|openEuler23.03|\
        almalinux8.7|almalinux8.8|almalinux9.2|\
        amzn2|amzn2022|amzn2023|\
        ol8.7|ol8.8|ol9.2|\
        rocky8.7|rocky8.8|rocky9.2|\
        fedora29|fedora30|fedora31|fedora32|fedora33|fedora34|fedora35|fedora36|fedora37|fedora38|fedora39|fedora40|\
        opensuse-leap15.3|\
        raspbian9|debian9|ubuntu18.04|\
        raspbian10|debian10|ubuntu20.04|ubuntu20.10|ubuntu21.04|\
        raspbian11|debian11|ubuntu21.10|ubuntu22.04|\
        raspbian12|debian12|ubuntu22.10) [ -z "$print" ] && echo "Detected OS: ${os}" ;;
        *) echo "Unsupported OS: ${os:-detection failed}" >&2 ; exit 1 ;;
    esac
fi

#===================================================================================================
# Collect packages

extra_repos=()

if [ "$os" == "raspbian9" ] || [ "$os" == "debian9" ] ; then

    # proper versions of cmake and python should be installed separately, because the defaults are:
    # - python version is 3.5
    # - cmake version is 3.7.2
    # which are not supported by OpenVINO

    pkgs_core=()
    pkgs_gpu=(ocl-icd-libopencl1)
    pkgs_python=()
    pkgs_dev=(pkg-config g++ gcc libc6-dev make sudo)

elif [ "$os" == "ubuntu18.04" ] ; then

    pkgs_core=(libtbb2)
    pkgs_gpu=(ocl-icd-libopencl1)
    pkgs_python=(python3.8 libpython3.8 python3.8-venv python3-pip)
    pkgs_dev=(cmake pkg-config g++ gcc libc6-dev make sudo)

elif [ "$os" == "ubuntu20.04" ] || [ "$os" == "debian10" ] || [ "$os" == "raspbian10" ] ||
     [ "$os" == "ubuntu21.10" ] || [ "$os" == "ubuntu22.04" ] || [ "$os" == "debian11" ] || [ "$os" == "raspbian11" ] ||
     [ "$os" == "ubuntu22.10" ] || [ "$os" == "ubuntu23.04" ] || [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ]; then

    pkgs_core=()
    pkgs_gpu=(ocl-icd-libopencl1)
    pkgs_python=(python3 python3-venv python3-pip)
    pkgs_dev=(cmake pkg-config g++ gcc libc6-dev make sudo)

    if [ "$os" == "ubuntu22.04" ] || [ "$os" == "ubuntu22.10" ] || [ "$os" == "ubuntu23.04" ] ||
       [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ] ; then
        pkgs_core+=(libtbb12)
    else
        pkgs_core+=(libtbb2)
    fi

    if [ "$os" == "debian10" ] || [ "$os" == "raspbian10" ] ; then
        pkgs_python+=(libpython3.7)
    elif [ "$os" == "ubuntu20.04" ] || [ "$os" == "ubuntu20.10" ] || [ "$os" == "ubuntu21.04" ] ; then
        pkgs_python+=(libpython3.8)
    elif [ "$os" == "ubuntu21.10" ] ||
         [ "$os" == "debian11" ] || [ "$os" == "raspbian11" ] ; then
        pkgs_python+=(libpython3.9)
    elif [ "$os" == "ubuntu22.04" ] || [ "$os" == "ubuntu22.10" ] ||
         [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ] ; then
        pkgs_python+=(libpython3.10)
    elif [ "$os" == "ubuntu23.04" ] ; then
        pkgs_python+=(libpython3.11)
    fi

elif [ "$os" == "centos7" ] || [ "$os" == "centos8" ] || [ "$os" == "centos9" ] ||
     [ "$os" == "rhel8" ] || [ "$os" == "rhel9.1" ] ||
     [ "$os" == "anolis8.6" ] || [ "$os" == "anolis8.8" ] ||
     [ "$os" == "openEuler20.03" ] || [ "$os" == "openEuler22.03" ] || [ "$os" == "openEuler23.03" ] ||
     [ "$os" == "fedora29" ] || [ "$os" == "fedora30" ] || [ "$os" == "fedora31" ] || [ "$os" == "fedora32" ] ||
     [ "$os" == "fedora33" ] || [ "$os" == "fedora34" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
     [ "$os" == "fedora37" ] || [ "$os" == "fedora38" ] || [ "$os" == "fedora39" ] || [ "$os" == "fedora40" ] ||
     [ "$os" == "ol8.7" ] || [ "$os" == "ol8.8" ] || [ "$os" == "ol9.2" ] ||
     [ "$os" == "rocky8.7" ] || [ "$os" == "rocky8.8" ] || [ "$os" == "rocky9.2" ] ||
     [ "$os" == "almalinux8.7" ] || [ "$os" == "almalinux8.8" ] || [ "$os" == "almalinux9.2" ] ||
     [ "$os" == "amzn2" ] || [ "$os" == "amzn2022" ] || [ "$os" == "amzn2023" ] ; then

    arch=$(uname -m)

    if [ "$os" == "amzn2" ] ; then
        amazon-linux-extras install epel python3.8
    fi

    pkgs_core=()
    pkgs_gpu=()
    pkgs_python=()
    pkgs_dev=(gcc gcc-c++ make glibc libstdc++ libgcc cmake3 sudo)

    if [ "$os" == "centos7" ] || [ "$os" == "amzn2" ] ; then
        pkgs_dev+=(pkgconfig)
    else
        pkgs_dev+=(pkgconf-pkg-config)
    fi

    if [ "$os" == "fedora29" ] || [ "$os" == "fedora30" ] || [ "$os" == "fedora31" ] || [ "$os" == "fedora32" ] ||
       [ "$os" == "fedora33" ] || [ "$os" == "fedora34" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
       [ "$os" == "fedora37" ] || [ "$os" == "fedora38" ] || [ "$os" == "fedora39" ] || [ "$os" == "fedora40" ] ||
       [ "$os" == "ol8.7" ] || [ "$os" == "ol8.8" ] || [ "$os" == "ol9.2" ] ||
       [ "$os" == "rocky8.7" ] || [ "$os" == "rocky8.8" ] || [ "$os" == "rocky9.2" ] ||
       [ "$os" == "almalinux8.7" ] || [ "$os" == "almalinux8.8" ] || [ "$os" == "almalinux9.2" ] ||
       [ "$os" == "centos8" ] || [ "$os" == "centos9" ] ||
       [ "$os" == "amzn2022" ] || [ "$os" == "amzn2023" ] ||
       [ "$os" == "anolis8.6" ] || [ "$os" == "anolis8.8" ] ||
       [ "$os" == "openEuler20.03" ] || [ "$os" == "openEuler22.03" ] || [ "$os" == "openEuler23.03" ] ; then
        pkgs_core+=("tbb.$arch")
        pkgs_python+=(python3 python3-pip)
    fi

    if [ "$os" == "centos7" ] || [ "$os" == "amzn2" ] ; then
        pkgs_gpu+=("ocl-icd.$arch")
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm")
    elif [ "$os" == "rhel8" ] ; then
        pkgs_core+=("https://vault.centos.org/centos/8/AppStream/$arch/os/Packages/tbb-2018.2-9.el8.$arch.rpm")
        pkgs_gpu+=("http://mirror.centos.org/centos/8-stream/AppStream/$arch/os/Packages/ocl-icd-2.2.12-1.el8.$arch.rpm")
        pkgs_python+=(python38 python38-pip)
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm")
    elif [ "$os" == "rhel9.1" ] ; then
        pkgs_core+=("http://mirror.stream.centos.org/9-stream/AppStream/$arch/os/Packages/tbb-2020.3-8.el9.$arch.rpm")
        pkgs_gpu+=("https://mirror.stream.centos.org/9-stream/AppStream/$arch/os/Packages/ocl-icd-2.2.13-4.el9.$arch.rpm")
        pkgs_python+=(python3 python3-pip)
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm")
    fi
elif [ "$os" == "opensuse-leap15.3" ] ; then
    pkgs_core=(libtbb2 libtbbmalloc2)
    pkgs_gpu=(libOpenCL1)
    pkgs_python=(python39-base python39 python39-venv python39-pip)
    pkgs_dev=(cmake pkg-config gcc-c++ gcc make sudo)
else
    echo "Internal script error: invalid OS (${os}) after check (package selection)" >&2
    exit 3
fi

#===================================================================================================
# Gather packages and print list

pkgs=()
for comp in "${comp[@]}" ; do
    var="pkgs_${comp}[@]"
    pkgs+=("${!var}")
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

if [ "$os" == "debian9" ] || [ "$os" == "raspbian9" ] || [ "$os" == "ubuntu18.04" ] ||
   [ "$os" == "debian10" ] || [ "$os" == "raspbian10" ] || [ "$os" == "ubuntu20.04" ] || [ "$os" == "ubuntu20.10" ] || [ "$os" == "ubuntu21.04" ] ||
   [ "$os" == "debian11" ] || [ "$os" == "raspbian11" ] || [ "$os" == "ubuntu21.10" ] || [ "$os" == "ubuntu22.04" ] ||
   [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ] || [ "$os" == "ubuntu22.10" ] ; then

    [ -z "$interactive" ] && iopt="-y"
    [ -n "$dry" ] && iopt="--dry-run"
    [ -n "$keepcache" ] && rm -f /etc/apt/apt.conf.d/docker-clean

    apt-get update && apt-get install --no-install-recommends "$iopt" "${pkgs[@]}"

elif [ "$os" == "centos7" ] || [ "$os" == "centos8" ] || [ "$os" == "centos9" ] ||
     [ "$os" == "rhel8" ] || [ "$os" == "rhel9.1" ] ||
     [ "$os" == "anolis8.6" ] || [ "$os" == "anolis8.8" ] ||
     [ "$os" == "openEuler20.03" ] || [ "$os" == "openEuler22.03" ] || [ "$os" == "openEuler23.03" ] ||
     [ "$os" == "fedora29" ] || [ "$os" == "fedora30" ] || [ "$os" == "fedora31" ] || [ "$os" == "fedora32" ] ||
     [ "$os" == "fedora33" ] || [ "$os" == "fedora34" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
     [ "$os" == "fedora37" ] || [ "$os" == "fedora38" ] || [ "$os" == "fedora39" ] || [ "$os" == "fedora40" ] ||
     [ "$os" == "fedora36" ] || [ "$os" == "fedora38" ] ||
     [ "$os" == "ol8.7" ] || [ "$os" == "ol8.8" ] || [ "$os" == "ol9.2" ] ||
     [ "$os" == "rocky8.7" ] || [ "$os" == "rocky8.8" ] || [ "$os" == "rocky9.2" ] ||
     [ "$os" == "almalinux8.7" ] || [ "$os" == "almalinux8.8" ] || [ "$os" == "almalinux9.2" ] ||
     [ "$os" == "amzn2" ] || [ "$os" == "amzn2022" ] || [ "$os" == "amzn2023" ] ; then

    [ -z "$interactive" ] && iopt="--assumeyes"
    [ -n "$dry" ] && iopt="--downloadonly"
    [ -n "$keepcache" ] && iopt="$iopt --setopt=keepcache=1"
    [ -n "$extra" ] && [ ${#extra_repos[@]} -ne 0 ] && yum localinstall "$iopt" --nogpgcheck "${extra_repos[@]}"

    yum install "$iopt" "${pkgs[@]}"

elif [ "$os" == "opensuse-leap15.3" ] ; then

    [ -z "$interactive" ] && iopt="-y"
    [ -n "$dry" ] && iopt="--dry-run"
    [ -n "$keepcache" ] && zypper clean --all

    zypper ref && zypper in --auto-agree-with-licenses --no-recommends "$iopt" "${pkgs[@]}"

else
    echo "Internal script error: invalid OS (${os}) after check (package installation)" >&2
    exit 3
fi

exit 0
