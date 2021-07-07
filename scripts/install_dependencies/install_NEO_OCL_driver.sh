#!/bin/bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Installs the Intel® Graphics Compute Runtime for OpenCL™ Driver on Linux.
#
# Usage: sudo -E ./install_NEO_OCL_driver.sh
#
# Supported platforms:
#     6th-11th generation Intel® Core™ processor with Intel(R)
#     Processor Graphics Technology not previously disabled by the BIOS
#     or motherboard settings
#
EXIT_FAILURE=1
EXIT_WRONG_ARG=2
CENTOS_MINOR=
RHEL_VERSION=
UBUNTU_VERSION=
DISTRO=
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"
INSTALL_DRIVER_VERSION='19.41.14441'
AVAILABLE_DRIVERS=("19.41.14441" "20.35.17767")


print_help()
{
    # Display Help
    usage="Usage: $(basename "$0") [OPTIONS]...
Download and installs the Intel® Graphics Compute Runtime for OpenCL™ Driver on Linux

    Available options:
    -y                      Replace the currently installed driver with the newer version.
    -a, --auto              Auto-mode for detecting best driver for current OS and hardware.
    -d, --install_driver    Manually select driver version to one of available to install drivers.
                            Default value: $INSTALL_DRIVER_VERSION
                            Available to install drivers: ${AVAILABLE_DRIVERS[*]}
    --no_numa               Skip installing NUMA packages. (off)
    -h, --help              Display this help and exit"
    echo "$usage"
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -d|--install_driver)
        user_chosen_driver="$2"
        if [[ " ${AVAILABLE_DRIVERS[*]} " =~ " ${user_chosen_driver} " ]]; then
            INSTALL_DRIVER_VERSION=$user_chosen_driver
        else
            echo "ERROR: unable to install the driver ${user_chosen_driver}."
            echo "Available values: ${AVAILABLE_DRIVERS[*]}"
            exit $EXIT_WRONG_ARG
        fi
        shift
        shift
    ;;
        -y)
        agreement=true
        shift
    ;;
        -a|--auto)
        auto_mode=true
        shift
    ;;
        --no_numa)
        no_numa=true
        shift
    ;;
        -h|--help)
        print_help
        exit
    ;;
        *)
        echo "$(basename "$0"): invalid option -- '${key}'"
        echo "Try '$(basename "$0") --help' for more information."
        exit $EXIT_WRONG_ARG
    esac
done

_install_prerequisites_redhat()
{
    # yum doesn't accept timeout in seconds as parameter
    echo
    echo "Note: if yum becomes non-responsive, try aborting the script and run:"
    echo "      sudo -E $0"
    echo

    CMDS=("yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && yum install -y ocl-icd")

    for cmd in "${CMDS[@]}"; do
        echo "$cmd"
        eval "$cmd"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: failed to run $cmd" >&2
            echo "Problem (or disk space)?" >&2
            echo ". Verify that you have enough disk space, and run the script again." >&2
            exit $EXIT_FAILURE
        fi
    done

}

_install_prerequisites_centos()
{
    # yum doesn't accept timeout in seconds as parameter
    echo
    echo "Note: if yum becomes non-responsive, try aborting the script and run:"
    echo "      sudo -E $0"
    echo

    if [ "$no_numa" == true ]; then
        CMDS=("yum install -y epel-release"
              "yum -y install ocl-icd ocl-icd-devel")
    else
        CMDS=("yum install -y epel-release"
              "yum -y install numactl-libs numactl ocl-icd ocl-icd-devel")
    fi

    for cmd in "${CMDS[@]}"; do
        echo "$cmd"
        eval "$cmd"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: failed to run $cmd" >&2
            echo "Problem (or disk space)?" >&2
            echo ". Verify that you have enough disk space, and run the script again." >&2
            exit $EXIT_FAILURE
        fi
    done

}

_install_prerequisites_ubuntu()
{
    if [ "$no_numa" == true ]; then
        CMDS=("apt-get -y update"
              "apt-get -y install --no-install-recommends ocl-icd-libopencl1")
    else
        CMDS=("apt-get -y update"
              "apt-get -y install --no-install-recommends libnuma1 ocl-icd-libopencl1")
    fi


    for cmd in "${CMDS[@]}"; do
        echo "$cmd"
        eval "$cmd"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: failed to run $cmd" >&2
            echo "Problem (or disk space)?" >&2
            echo "                sudo -E $0" >&2
            echo "2. Verify that you have enough disk space, and run the script again." >&2
            exit $EXIT_FAILURE
        fi
    done
}

install_prerequisites()
{
    echo "Installing prerequisites..."
    if [[ $DISTRO == "centos" ]]; then
        _install_prerequisites_centos
    elif [[ $DISTRO == "redhat" ]]; then
        _install_prerequisites_redhat
    elif [[ $DISTRO == "ubuntu" ]]; then
        _install_prerequisites_ubuntu
    else
        echo Unknown OS
    fi
}

_deploy_rpm()
{
    # On a CentOS 7.2 machine with Intel Parallel Composer XE 2017
    # installed we got conflicts when trying to deploy these rpms.
    # If that happens to you too, try again with:
    # IGFX_RPM_FLAGS="--force" sudo -E ./install_NEO_OCL_driver.sh install
    #
    cmd="rpm $IGFX_RPM_FLAGS -ivh --nodeps --force $1"
    echo "$cmd"
    eval "$cmd"
}

_deploy_deb()
{
    cmd="dpkg -i $1"
    echo "$cmd"
    eval "$cmd"
}

_install_user_mode_centos()
{
    _deploy_rpm "intel*.rpm"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: failed to install rpms $cmd error"  >&2
        echo "Make sure you have enough disk space or fix the problem manually and try again." >&2
        exit $EXIT_FAILURE
    fi
}

_install_user_mode_ubuntu()
{
    _deploy_deb "intel*.deb"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: failed to install rpms $cmd error"  >&2
        echo "Make sure you have enough disk space or fix the problem manually and try again." >&2
        exit $EXIT_FAILURE
    fi
}


install_user_mode()
{
    echo "Installing user mode driver..."
    
    if [[ $DISTRO == "centos" || $DISTRO == "redhat" ]]; then
        _install_user_mode_centos
    else
        _install_user_mode_ubuntu
    fi
    # exit from $SCRIPT_DIR/neo folder
    cd - || exit
    # clean it up
    rm -rf "$SCRIPT_DIR/neo"
}

_uninstall_user_mode_centos()
{
    echo Looking for previously installed user-mode driver...
    PACKAGES=("intel-opencl"
           "intel-ocloc"
           "intel-gmmlib"
           "intel-igc-core"
           "intel-igc-opencl")
    for package in "${PACKAGES[@]}"; do      
        echo "rpm -qa | grep $package"
        found_package=$(rpm -qa | grep "$package")
        if [[ $? -eq 0 ]]; then
            echo "Found installed user-mode driver, performing uninstall..."
            cmd="rpm -e --nodeps ${found_package}"
            echo "$cmd"
            eval "$cmd"
            if [[ $? -ne 0 ]]; then
                echo "ERROR: failed to uninstall existing user-mode driver." >&2
                echo "Please try again manually and run the script again." >&2
                exit $EXIT_FAILURE
            fi
        fi
    done
}

_uninstall_user_mode_ubuntu()
{
    echo Looking for previously installed user-mode driver...

    PACKAGES=("intel-opencl"
              "intel-ocloc"
              "intel-gmmlib"
              "intel-igc-core"
              "intel-igc-opencl")

    for package in "${PACKAGES[@]}"; do
        found_package=$(dpkg-query -W -f='${binary:Package}\n' "${package}")
        if [[ $? -eq 0 ]]; then
            echo "Found installed user-mode driver, performing uninstall..."
            cmd="apt-get autoremove -y $package"
            echo "$cmd"
            eval "$cmd"
            if [[ $? -ne 0 ]]; then
                echo "ERROR: failed to uninstall existing user-mode driver." >&2
                echo "Please try again manually and run the script again." >&2
                exit $EXIT_FAILURE
            fi
        fi
    done
}

uninstall_user_mode()
{
    if [[ $DISTRO == "centos" || $DISTRO == "redhat" ]]; then
        _uninstall_user_mode_centos
    else
        _uninstall_user_mode_ubuntu
    fi
}

_is_package_installed()
{
    if [[ $DISTRO == "centos" || $DISTRO == "redhat" ]]; then
        cmd="rpm -qa | grep $1"
    else
        cmd="dpkg-query -W -f='${binary:Package}\n' $pkg"
    fi
    echo "$cmd"
    eval "$cmd"
}


_download_packages_ubuntu()
{
    case $INSTALL_DRIVER_VERSION in
    "19.41.14441")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.41.14441_amd64.deb
    ;;
    "20.35.17767")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-gmmlib_20.2.4_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-igc-core_1.0.4756_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-igc-opencl_1.0.4756_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-opencl_20.35.17767_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-ocloc_20.35.17767_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/intel-level-zero-gpu_1.0.17767_amd64.deb
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        echo "Available values: ${AVAILABLE_DRIVERS[*]}"
        exit $EXIT_WRONG_ARG
    esac
}

_download_packages_centos()
{

    case $INSTALL_DRIVER_VERSION in
    "19.41.14441")
        curl -L --output intel-igc-core-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-core-1.0.2597-1.el7.x86_64.rpm/download
        curl -L --output intel-opencl-19.41.14441-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-opencl-19.41.14441-1.el7.x86_64.rpm/download
        curl -L --output intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm/download
        curl -L --output intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm/download
        curl -L --output intel-gmmlib-19.3.2-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-19.3.2-1.el7.x86_64.rpm/download
        curl -L --output intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm/download
    ;;
    "20.35.17767")
        curl -L --output intel-opencl-20.35.17767-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-opencl-20.35.17767-1.el7.x86_64.rpm/download
        curl -L --output level-zero-1.0.0-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/level-zero-1.0.0-1.el7.x86_64.rpm/download
        curl -L --output level-zero-devel-1.0.0-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/level-zero-devel-1.0.0-1.el7.x86_64.rpm/download
        curl -L --output intel-igc-opencl-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-opencl-1.0.4756-1.el7.x86_64.rpm/download
        curl -L --output intel-igc-opencl-devel-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-opencl-devel-1.0.4756-1.el7.x86_64.rpm/download
        curl -L --output intel-igc-core-1.0.4756-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-igc-core-1.0.4756-1.el7.x86_64.rpm/download
        curl -L --output intel-gmmlib-20.2.4-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-gmmlib-20.2.4-1.el7.x86_64.rpm/download
        curl -L --output intel-gmmlib-devel-20.2.4-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/20.35.17767/centos-7/intel-gmmlib-devel-20.2.4-1.el7.x86_64.rpm/download
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        echo "Available values: ${AVAILABLE_DRIVERS[*]}"
        exit $EXIT_WRONG_ARG
    esac
}


_verify_checksum_ubuntu()
{
    case $INSTALL_DRIVER_VERSION in
    "19.41.14441")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/19.41.14441/ww41.sum
        sha256sum -c ww41.sum
    ;;
    "20.35.17767")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/20.35.17767/ww35.sum
        sha256sum -c ww35.sum
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        echo "Available values: ${AVAILABLE_DRIVERS[*]}"
        exit $EXIT_WRONG_ARG
    esac
}

_verify_checksum_centos()
{
    case $INSTALL_DRIVER_VERSION in
    "19.41.14441")
        sha1sum -c "$SCRIPT_DIR/neo_centos_19.41.14441.sum"
    ;;
    "20.35.17767")
        sha1sum -c "$SCRIPT_DIR/neo_centos_20.35.17767.sum"
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        echo "Available values: ${AVAILABLE_DRIVERS[*]}"
        exit $EXIT_WRONG_ARG
    esac    
}

verify_checksum()
{
    if [[ $DISTRO == "centos" || $DISTRO == "redhat" ]]; then
        _verify_checksum_centos
    else
        _verify_checksum_ubuntu
    fi
}

download_packages()
{
    mkdir -p "$SCRIPT_DIR/neo"
    cd "$SCRIPT_DIR/neo" || exit
    
    if [[ $DISTRO == "centos" || $DISTRO == "redhat" ]]; then
        _download_packages_centos
    else
        _download_packages_ubuntu
    fi
    verify_checksum
    if [[ $? -ne 0 ]]; then
        echo "ERROR: checksums do not match for the downloaded packages"
        echo "       Please verify your Internet connection and make sure you have enough disk space or fix the problem manually and try again. "
        exit $EXIT_FAILURE
    fi
}


version_gt() {
    # check if first version is greater than second version
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}

summary()
{
    echo
    echo "Installation completed successfully."
    echo
    echo "Next steps:"
    echo "Add OpenCL users to the video and render group: 'sudo usermod -a -G video,render USERNAME'"
    echo "   e.g. if the user running OpenCL host applications is foo, run: sudo usermod -a -G video,render foo"
    echo "   Current user has been already added to the video and render group"
    echo

    echo "If you use 8th Generation Intel® Core™ processor, add:"
    echo "   i915.alpha_support=1"
    echo "   to the 4.14 kernel command line, in order to enable OpenCL functionality for this platform."
    echo
 
}

check_root_access()
{
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: you must run this script as root." >&2
        echo "Please try again with \"sudo -E $0\", or as root." >&2
        exit $EXIT_FAILURE
    fi
}

add_user_to_video_group()
{
    local real_user
    real_user=$(logname 2>/dev/null || echo "${SUDO_USER:-${USER}}")
    echo
    echo "Adding $real_user to the video group..."
    usermod -a -G video "$real_user"
    if [[ $? -ne 0 ]]; then
        echo "WARNING: unable to add $real_user to the video group" >&2
    fi
    echo "Adding $real_user to the render group..."
    usermod -a -G render "$real_user"
    if [[ $? -ne 0 ]]; then
        echo "WARNING: unable to add $real_user to the render group" >&2
    fi
}

_check_distro_version()
{
    if [[ $DISTRO == centos ]]; then
        CENTOS_MINOR=$(sed 's/CentOS Linux release 7\.\([[:digit:]]\+\).\+/\1/' /etc/centos-release)
        if [[ $? -ne 0 ]]; then
            echo "ERROR: failed to obtain CentOS version minor." >&2
            echo "This script is supported only on CentOS 7 and above." >&2
            exit $EXIT_FAILURE
        fi
    elif [[ $DISTRO == redhat ]]; then
        RHEL_VERSION=$(grep -m1 'VERSION_ID' /etc/os-release | grep -Eo "8.[0-9]")
        if [[ $? -ne 0 ]]; then
            echo "Warning: This runtime can be installed only on RHEL 8" >&2
            echo "Installation of Intel Compute Runtime interrupted"
            exit $EXIT_FAILURE
        fi
    elif [[ $DISTRO == ubuntu ]]; then
        UBUNTU_VERSION=$(grep -m1 'VERSION_ID' /etc/os-release | grep -Eo "[0-9]{2}.[0-9]{2}") 
        if [[ $UBUNTU_VERSION != '18.04' && $UBUNTU_VERSION != '20.04' ]]; then
            echo "Warning: This runtime can be installed only on Ubuntu 18.04 or Ubuntu 20.04."
            echo "More info https://github.com/intel/compute-runtime/releases" >&2
            echo "Installation of Intel Compute Runtime interrupted"
            exit $EXIT_FAILURE
        fi
    fi
}

distro_init()
{
    if [[ -f /etc/centos-release ]]; then
        DISTRO="centos"
    elif [[ -f /etc/redhat-release ]]; then
        DISTRO="redhat"
    elif [[ -f /etc/lsb-release ]]; then
        DISTRO="ubuntu"
    fi

    _check_distro_version
}

check_agreement()
{
    if [ "$agreement" == true ]; then
        return 0
    fi

    echo "This script will download and install Intel(R) Graphics Compute Runtime $INSTALL_DRIVER_VERSION, "
    echo "that was used to validate this OpenVINO™ package."
    echo "In case if you already have the driver - script will try to remove it."
    while true; do
        read -p "Want to proceed? (y/n): " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) exit $EXIT_FAILURE ;;
        esac
    done
}

check_specific_generation()
{
    echo "Checking processor generation..."
    specific_generation=$(grep -m1 'model name' /proc/cpuinfo | grep -E "i[357]-1[01][0-9]{2,4}N?G[147R]E?")
    if [[ ! -z "$specific_generation" && "$INSTALL_DRIVER_VERSION" != '20.35.17767' ]]; then
        echo "$(basename "$0"): Detected 10th generation Intel® Core™ processor (formerly Ice Lake) or 11th generation Intel® Core™ processor (formerly Tiger Lake)."
        echo "Driver version 20.35.17767 is going to be installed to fully utilize hardware features and performance."
        if [ "$auto_mode" == true ]; then
            INSTALL_DRIVER_VERSION='20.35.17767'
            return 0
        else
            while true; do
                read -p "You are still able to use the older version 19.41.14441. Use the older driver? (y/n) [n] " yn
                yn=${yn:=n}
                case $yn in
                    [Yy]*) return 0 ;;
                    [Nn]*) INSTALL_DRIVER_VERSION='20.35.17767' && return 0 ;;
                esac
            done
        fi
    fi
}

check_current_driver()
{   
    echo "Checking current driver version..."
    if [[ $DISTRO == centos || $DISTRO == redhat ]]; then
        gfx_version=$(yum info intel-opencl | grep Version)
    elif [[ $DISTRO == ubuntu ]]; then
        gfx_version=$(apt show intel-opencl | grep Version)
    fi
    
    gfx_version="$(echo -e "${gfx_version}" | sed -e 's/^Version[[:space:]]*\:[[:space:]]*//')"
    check_specific_generation
    
    # install NEO OCL driver if the current driver version < INSTALL_DRIVER_VERSION
    if [[ ! -z $gfx_version && "$(printf '%s\n' "$INSTALL_DRIVER_VERSION" "$gfx_version" | sort -V | head -n 1)" = "$INSTALL_DRIVER_VERSION" ]]; then
        echo "Intel® Graphics Compute Runtime for OpenCL™ Driver installation skipped because current version greater or equal to $INSTALL_DRIVER_VERSION" >&2
        echo "Installation of Intel® Graphics Compute Runtime for OpenCL™ Driver interrupted." >&2
        exit $EXIT_FAILURE
    else
        echo "Starting installation..."
    fi
}

install()
{   
    uninstall_user_mode
    install_prerequisites
    download_packages
    install_user_mode
    add_user_to_video_group
}

main()
{
    echo "Intel® Graphics Compute Runtime for OpenCL™ Driver installer"
    distro_init
    check_root_access
    check_current_driver
    check_agreement
    install
    summary
}

[[ "$0" == "${BASH_SOURCE[0]}" ]] && main "$@"
