#!/bin/bash

# Copyright (C) 2018-2023 Intel Corporation
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
    for image in centos7 centos8 rhel8 rhel9.1 \
                 almalinux8.7 amzn2 \
                 fedora34 fedora35 fedora36 fedora37 fedora38 \
                 raspbian9 debian9 ubuntu18.04 \
                 raspbian10 debian10 ubuntu20.04 ubuntu20.10 ubuntu21.04 \
                 raspbian11 debian11 ubuntu21.10 ubuntu22.04 \
                 raspbian12 debian12 ubuntu22.10 ; do
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
        centos7|centos8|rhel8|rhel9.1|\
        almalinux8.7|amzn2|\
        opensuse-leap15.3| \
        fedora34|fedora35|fedora36|fedora37|fedora38|\
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

    pkgs_core=(libpugixml1v5)
    pkgs_gpu=()
    pkgs_python=()
    pkgs_dev=(pkg-config g++ gcc libc6-dev libgflags-dev zlib1g-dev nlohmann-json-dev make curl sudo)

elif [ "$os" == "ubuntu18.04" ] ; then

    pkgs_core=(libtbb2 libpugixml1v5)
    pkgs_gpu=()
    pkgs_python=(python3.8 libpython3.8 python3.8-venv python3-pip)
    pkgs_dev=(cmake pkg-config g++ gcc libc6-dev libgflags-dev zlib1g-dev nlohmann-json-dev make curl sudo)

elif [ "$os" == "ubuntu20.04" ] || [ "$os" == "debian10" ] || [ "$os" == "raspbian10" ] ||
     [ "$os" == "ubuntu21.10" ] || [ "$os" == "ubuntu22.04" ] || [ "$os" == "debian11" ] || [ "$os" == "raspbian11" ] ||
     [ "$os" == "ubuntu22.10" ] || [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ]; then

    pkgs_core=(libpugixml1v5 libtbb2)
    pkgs_gpu=()
    pkgs_python=(python3 python3-venv python3-pip)
    pkgs_dev=(cmake pkg-config g++ gcc libc6-dev libgflags-dev zlib1g-dev nlohmann-json3-dev make curl sudo)

    if [ "$os" == "debian10" ] || [ "$os" == "raspbian10" ] ; then
        pkgs_python=("${pkgs_python[@]}" libpython3.7)
    elif [ "$os" == "ubuntu20.04" ] || [ "$os" == "ubuntu20.10" ] || [ "$os" == "ubuntu21.04" ] ; then
        pkgs_python=("${pkgs_python[@]}" libpython3.8)
    elif [ "$os" == "ubuntu21.10" ] ||
         [ "$os" == "debian11" ] || [ "$os" == "raspbian11" ] ; then
        pkgs_python=("${pkgs_python[@]}" libpython3.9)
    elif [ "$os" == "ubuntu22.04" ] || [ "$os" == "ubuntu22.10" ] ||
         [ "$os" == "debian12" ] || [ "$os" == "raspbian12" ] ; then
        pkgs_python=("${pkgs_python[@]}" libpython3.10)
    fi

elif [ "$os" == "centos7" ] || [ "$os" == "centos8" ] ||
     [ "$os" == "rhel8" ] || [ "$os" == "rhel9.1" ] ||
     [ "$os" == "fedora34" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
     [ "$os" == "fedora36" ] || [ "$os" == "fedora38" ] ||
     [ "$os" == "almalinux8.7" ] || [ "$os" == "amzn2" ] ; then

    arch=$(uname -m)

    if [ "$os" == "amzn2" ] ; then
        amazon-linux-extras install epel python3.8
    fi

    pkgs_dev=(gcc gcc-c++ make glibc libstdc++ libgcc cmake3 "json-devel.$arch" "zlib-devel.$arch" sudo)
    pkgs_gpu=()

    if [ "$os" == "centos7" ] || [ "$os" == "amzn2" ] ; then
        pkgs_dev+=(pkgconfig)
    else
        pkgs_dev+=(pkgconf-pkg-config)
    fi

    if [ "$os" == "rhel9.1" ] ; then
        pkgs_dev+=(curl-minimal)
    else
        pkgs_dev+=(curl)
    fi

    if [ "$os" == "fedora35" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
       [ "$os" == "fedora36" ] || [ "$os" == "fedora38" ] ; then
        pkgs_core=("tbb.$arch" "pugixml.$arch" "gflags.$arch")
        pkgs_python=(python3 python3-pip)
        pkgs_dev+=("gflags-devel.$arch")
    fi

    if [ "$os" == "centos7" ] || [ "$os" == "amzn2" ] ; then
        pkgs_core=("tbb.$arch" "pugixml.$arch" "gflags.$arch")
        pkgs_dev+=("gflags-devel.$arch")
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm")
    elif [ "$os" == "centos8" ] || [ "$os" == "rhel8" ] || [ "$os" == "almalinux8.7" ] ; then
        pkgs_core+=(
            "https://vault.centos.org/centos/8/AppStream/$arch/os/Packages/tbb-2018.2-9.el8.$arch.rpm"
            "https://download-ib01.fedoraproject.org/pub/epel/8/Everything/$arch/Packages/p/pugixml-1.13-1.el8.$arch.rpm"
            "https://vault.centos.org/centos/8/PowerTools/$arch/os/Packages/gflags-2.1.2-6.el8.$arch.rpm"
        )
        pkgs_gpu+=(
            "http://mirror.centos.org/centos/8-stream/AppStream/x86_64/os/Packages/ocl-icd-2.2.12-1.el8.x86_64.rpm"
        )
        pkgs_python+=(python38 python38-pip)
        pkgs_dev+=(
            "https://vault.centos.org/centos/8/PowerTools/$arch/os/Packages/gflags-devel-2.1.2-6.el8.$arch.rpm"
            "https://download-ib01.fedoraproject.org/pub/epel/8/Everything/$arch/Packages/j/json-devel-3.6.1-2.el8.$arch.rpm"
        )
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm")
    elif [ "$os" == "rhel9.1" ] ; then
        pkgs_core=(
            "http://mirror.stream.centos.org/9-stream/AppStream/$arch/os/Packages/tbb-2020.3-8.el9.$arch.rpm"
            "https://download-ib01.fedoraproject.org/pub/epel/9/Everything/$arch/Packages/p/pugixml-1.13-1.el9.$arch.rpm"
            "https://download-ib01.fedoraproject.org/pub/epel/9/Everything/$arch/Packages/g/gflags-2.2.2-9.el9.$arch.rpm"
        )
        pkgs_python=(python3 python3-pip)
        pkgs_dev+=("https://download-ib01.fedoraproject.org/pub/epel/9/Everything/$arch/Packages/g/gflags-devel-2.2.2-9.el9.$arch.rpm")
        extra_repos+=("https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm")
    fi
elif [ "$os" == "opensuse-leap15.3" ] ; then
    pkgs_core=(libtbb2 libtbbmalloc2 libpugixml1)
    pkgs_gpu=()
    pkgs_python=(python39-base python39 python39-venv python39-pip)
    pkgs_dev=(cmake pkg-config gcc-c++ gcc gflags-devel-static zlib-devel nlohmann_json-devel make curl sudo)
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

elif [ "$os" == "centos7" ] || [ "$os" == "centos8" ] ||
     [ "$os" == "rhel8" ] || [ "$os" == "rhel9.1" ] ||
     [ "$os" == "fedora34" ] || [ "$os" == "fedora35" ] || [ "$os" == "fedora36" ] ||
     [ "$os" == "fedora36" ] || [ "$os" == "fedora38" ] ||
     [ "$os" == "almalinux8.7" ] || [ "$os" == "amzn2" ] ; then

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
