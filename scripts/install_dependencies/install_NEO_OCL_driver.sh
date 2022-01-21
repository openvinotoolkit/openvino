#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Installs the Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver on Linux.
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
RHEL_VERSION=
UBUNTU_VERSION=
DISTRO=
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"
INSTALL_DRIVER_VERSION='unknown'


print_help()
{
    # Display Help
    usage="Usage: $(basename "$0") [OPTIONS]...
Download and installs the Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver on Linux

    Available options:
    -y                      Replace the currently installed driver with the newer version.
    --no_numa               Skip installing NUMA packages. (off)
    -h, --help              Display this help and exit"
    echo "$usage"
}

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -d|--install_driver)
        echo "WARNING: This option is deprecated. Recommended driver for current platform will be installed."
        shift
        shift
    ;;
        -y)
        agreement=true
        shift
    ;;
        -a|--auto)
        echo "WARNING: This option is deprecated. Recommended driver for current platform will be installed."
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
    CMDS=("dnf install -y 'dnf-command(config-manager)'"
          "dnf config-manager --add-repo \
           https://repositories.intel.com/graphics/rhel/${RHEL_VERSION}/intel-graphics.repo")
    
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
    echo 'Installing prerequisites...'
    if [[ $DISTRO == "redhat" ]]; then
        _install_prerequisites_redhat
    elif [[ $DISTRO == "ubuntu" ]]; then
        _install_prerequisites_ubuntu
    else
        echo 'WARNING::install_prerequisites: Unknown OS'
    fi
}

_deploy_rpm()
{
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

_install_user_mode_redhat()
{   
    CMDS=("dnf install --refresh -y intel-igc-opencl-1.0.9441-i643.el8.x86_64 \
           intel-media-21.4.1-i643.el8.x86_64 \
           level-zero-1.6.2-i643.el8.x86_64 \
           intel-level-zero-gpu-1.2.21786-i643.el8.x86_64 \
           intel-opencl-21.49.21786-i643.el8.x86_64 \
           intel-igc-core-1.0.9441-i643.el8.x86_64 \
           intel-ocloc-21.49.21786-i643.el8.x86_64 \
           ocl-icd-2.2.12-1.el8.x86_64 \
           intel-gmmlib-21.3.3-i643.el8.x86_64")

    for cmd in "${CMDS[@]}"; do
        echo "$cmd"
        eval "$cmd"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: failed to run $cmd" >&2
            echo "Problem (or disk space)?" >&2
            echo "                sudo -E $0" >&2
            echo "Verify that you have enough disk space, and run the script again." >&2
            exit $EXIT_FAILURE
        fi
    done
}

_install_user_mode_ubuntu()
{
    _deploy_deb "intel*.deb"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: failed to install debs $cmd error"  >&2
        echo "Make sure you have enough disk space or fix the problem manually and try again." >&2
        exit $EXIT_FAILURE
    fi
}


install_user_mode()
{
    echo "Installing user mode driver..."
    
    if [[ $DISTRO == "redhat" ]]; then
        _install_user_mode_redhat
    else
        _install_user_mode_ubuntu
    fi
    # exit from $SCRIPT_DIR/neo folder
    cd - || exit
    # clean it up
    rm -rf "$SCRIPT_DIR/neo"
}

_uninstall_user_mode_redhat()
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
              "intel-opencl-icd"
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
    if [[ $DISTRO == "redhat" ]]; then
        _uninstall_user_mode_redhat
    else
        _uninstall_user_mode_ubuntu
    fi
}

_download_packages_ubuntu()
{
    case $INSTALL_DRIVER_VERSION in
    "21.38.21026")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.38.21026/intel-gmmlib_21.2.1_amd64.deb
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.8708/intel-igc-core_1.0.8708_amd64.deb
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.8708/intel-igc-opencl_1.0.8708_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.38.21026/intel-opencl_21.38.21026_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.38.21026/intel-ocloc_21.38.21026_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.38.21026/intel-level-zero-gpu_1.2.21026_amd64.deb
    ;;
    "21.48.21782")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-gmmlib_21.3.3_amd64.deb
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-core_1.0.9441_amd64.deb
        curl -L -O https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.9441/intel-igc-opencl_1.0.9441_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-opencl-icd_21.48.21782_amd64.deb
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/intel-level-zero-gpu_1.2.21782_amd64.deb
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        exit $EXIT_WRONG_ARG
    esac
}

_verify_checksum_ubuntu()
{
    case $INSTALL_DRIVER_VERSION in
    "21.38.21026")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.38.21026/ww38.sum
        sha256sum -c ww38.sum
    ;;
    "21.48.21782")
        curl -L -O https://github.com/intel/compute-runtime/releases/download/21.48.21782/ww48.sum
        sha256sum -c ww48.sum
    ;;
        *)
        echo "ERROR: Unrecognized driver ${INSTALL_DRIVER_VERSION}."
        exit $EXIT_WRONG_ARG
    esac
}

verify_checksum()
{
    if [[ $DISTRO == "redhat" ]]; then
        return 0
    else
        _verify_checksum_ubuntu
    fi
}

download_packages()
{
    mkdir -p "$SCRIPT_DIR/neo"
    cd "$SCRIPT_DIR/neo" || exit
    
    if [[ $DISTRO == "redhat" ]]; then
        return 0
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
    if [[ $DISTRO == redhat ]]; then
        RHEL_MINOR_VERSION_SUPPORTED="[3-4]"
        RHEL_VERSION=$(grep -m1 'VERSION_ID' /etc/os-release | grep -Eo "8.${RHEL_MINOR_VERSION_SUPPORTED}")
        if [[ $? -ne 0 ]]; then
            echo "Warning: This runtime can be installed only on RHEL 8.3 or RHEL 8.4"
            echo "More info https://dgpu-docs.intel.com/releases/releases-20211130.html" >&2
            echo "Installation of Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver interrupted"
            exit $EXIT_FAILURE
        else
            INSTALL_DRIVER_VERSION='21.49.21786'
        fi
    elif [[ $DISTRO == ubuntu ]]; then
        UBUNTU_VERSION=$(grep -m1 'VERSION_ID' /etc/os-release | grep -Eo "[0-9]{2}.[0-9]{2}") 
        if [[ $UBUNTU_VERSION == '18.04' ]]; then
            INSTALL_DRIVER_VERSION='21.38.21026'
        elif [[ $UBUNTU_VERSION == '20.04' ]]; then
            INSTALL_DRIVER_VERSION='21.48.21782'
        else
            echo "Warning: This runtime can be installed only on Ubuntu 18.04 or Ubuntu 20.04."
            echo "More info https://github.com/intel/compute-runtime/releases" >&2
            echo "Installation of Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver interrupted"
            exit $EXIT_FAILURE
        fi
    fi
}

distro_init()
{
    if [[ -f /etc/redhat-release ]]; then
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

    echo "This script will download and install Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver $INSTALL_DRIVER_VERSION, "
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

check_current_driver()
{   
    echo "Checking current driver version..."
    if [[ $DISTRO == redhat ]]; then
        gfx_version=$(yum info intel-opencl | grep Version)
    elif [[ $DISTRO == ubuntu ]]; then
        gfx_version=$(dpkg-query --showformat='${Version}' --show intel-opencl)
        if [[ -z "$gfx_version" ]]; then 
            gfx_version=$(dpkg-query --showformat='${Version}' --show intel-opencl-icd)
        fi
    fi
    
    gfx_version="$(echo -e "${gfx_version}" | grep -Eo "[0-9]{2,3}\.[0-9]{2,3}\.[0-9]{3,6}")"
    
    # install NEO OCL driver if the current driver version < INSTALL_DRIVER_VERSION
    if [[ ! -z $gfx_version && "$(printf '%s\n' "$INSTALL_DRIVER_VERSION" "$gfx_version" | sort -V | head -n 1)" = "$INSTALL_DRIVER_VERSION" ]]; then
        echo "Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver installation skipped because current version greater or equal to $INSTALL_DRIVER_VERSION" >&2
        echo "Installation of Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver interrupted." >&2
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
    echo "Intel® Graphics Compute Runtime for oneAPI Level Zero and OpenCL™ Driver installer"
    distro_init
    check_root_access
    check_current_driver
    check_agreement
    install
    summary
}

[[ "$0" == "${BASH_SOURCE[0]}" ]] && main "$@"
