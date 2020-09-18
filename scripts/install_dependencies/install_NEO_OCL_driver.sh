#!/bin/bash

# Copyright (c) 2018 - 2020 Intel Corporation
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

#
# Installs the Graphics Driver for OpenCL on Linux.
#
# Usage: sudo -E ./install_NEO_OCL_driver.sh
#
# Supported platforms:
#     6th, 7th, 8th or 9th generation Intel® processor with Intel(R)
#     Processor Graphics Technology not previously disabled by the BIOS
#     or motherboard settings
#
EXIT_FAILURE=1
PKGS=
CENTOS_MINOR=
UBUNTU_VERSION=
DISTRO=
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
INSTALL_DRIVER_VERSION='19.41.14441'

params=$@

_install_prerequisites_centos()
{
    # yum doesn't accept timeout in seconds as parameter
    echo
    echo "Note: if yum becomes non-responsive, try aborting the script and run:"
    echo "      sudo -E $0"
    echo

    CMDS=("yum -y install tar libpciaccess numactl-libs"
          "yum -y groupinstall 'Development Tools'"
          "yum -y install rpmdevtools openssl openssl-devel bc numactl ocl-icd ocl-icd-devel")

    for cmd in "${CMDS[@]}"; do
        echo $cmd
        eval $cmd
        if [[ $? -ne 0 ]]; then
            echo ERROR: failed to run $cmd >&2
            echo Problem \(or disk space\)? >&2
            echo . Verify that you have enough disk space, and run the script again. >&2
            exit $EXIT_FAILURE
        fi
    done

}

_install_prerequisites_ubuntu()
{
    CMDS=("apt-get -y update"
          "apt-get -y install libnuma1 ocl-icd-libopencl1")

    for cmd in "${CMDS[@]}"; do
        echo $cmd
        eval $cmd
        if [[ $? -ne 0 ]]; then
            echo ERROR: failed to run $cmd >&2
            echo Problem \(or disk space\)? >&2
            echo "                sudo -E $0" >&2
            echo 2. Verify that you have enough disk space, and run the script again. >&2
            exit $EXIT_FAILURE
        fi
    done
}

install_prerequisites()
{
    if [[ $DISTRO == "centos" ]]; then
        echo Installing prerequisites...
        _install_prerequisites_centos
    elif [[ $DISTRO == "ubuntu" ]]; then
        echo Installing prerequisites...
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
    echo $cmd
    eval $cmd
}

_deploy_deb()
{
    cmd="dpkg -i $1"
    echo $cmd
    eval $cmd
}


_install_user_mode_centos()
{
    _deploy_rpm "intel*.rpm"
    if [[ $? -ne 0 ]]; then
        echo ERROR: failed to install rpms $cmd error  >&2
        echo Make sure you have enough disk space or fix the problem manually and try again. >&2
        exit $EXIT_FAILURE
    fi
}

_install_user_mode_ubuntu()
{
    _deploy_deb "intel*.deb"
    if [[ $? -ne 0 ]]; then
        echo ERROR: failed to install rpms $cmd error  >&2
        echo Make sure you have enough disk space or fix the problem manually and try again. >&2
        exit $EXIT_FAILURE
    fi
}


install_user_mode()
{
    echo "Installing user mode driver..."
    
    if [[ $DISTRO == "centos" ]]; then
        _install_user_mode_centos
    else
        _install_user_mode_ubuntu
    fi
    # exit from $SCRIPT_DIR/neo folder
    cd -
    # clean it up
    rm -rf $SCRIPT_DIR/neo
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
        found_package=$(rpm -qa | grep $package)
        if [[ $? -eq 0 ]]; then
            echo Found installed user-mode driver, performing uninstall...
            cmd="rpm -e --nodeps ${found_package}"
            echo $cmd
            eval $cmd
            if [[ $? -ne 0 ]]; then
                echo ERROR: failed to uninstall existing user-mode driver. >&2
                echo Please try again manually and run the script again. >&2
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
        apt remove -y $package
        if [[ $? -ne 0 ]]; then
            echo ERROR: failed to uninstall existing user-mode driver. >&2
            echo Please try again manually and run the script again. >&2
            exit $EXIT_FAILURE
        fi
    done
}

uninstall_user_mode()
{
    if [[ $DISTRO == "centos" ]]; then
        _uninstall_user_mode_centos
    else
        _uninstall_user_mode_ubuntu
    fi
}

_is_package_installed()
{
    if [[ $DISTRO == "centos" ]]; then
        cmd="rpm -qa | grep $1"
    else
        cmd="dpkg-query -W -f='${binary:Package}\n' $pkg"
    fi
    echo $cmd
    eval $cmd
}


_download_packages_ubuntu()
{
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-gmmlib_19.3.2_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-core_1.0.2597_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-igc-opencl_1.0.2597_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-opencl_19.41.14441_amd64.deb
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/intel-ocloc_19.41.14441_amd64.deb
}

_download_packages_centos()
{
    wget -O intel-igc-core-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-core-1.0.2597-1.el7.x86_64.rpm/download
    wget -O intel-opencl-19.41.14441-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-opencl-19.41.14441-1.el7.x86_64.rpm/download
    wget -O intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-devel-1.0.2597-1.el7.x86_64.rpm/download
    wget -O intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-igc-opencl-1.0.2597-1.el7.x86_64.rpm/download
    wget -O intel-gmmlib-19.3.2-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-19.3.2-1.el7.x86_64.rpm/download
    wget -O intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm https://sourceforge.net/projects/intel-compute-runtime/files/19.41.14441/centos-7/intel-gmmlib-devel-19.3.2-1.el7.x86_64.rpm/download
}


_verify_checksum_ubuntu()
{
    wget https://github.com/intel/compute-runtime/releases/download/19.41.14441/ww41.sum
    sha256sum -c ww41.sum
}

_verify_checksum_centos()
{
    sha1sum -c $SCRIPT_DIR/neo_centos_ww41.sum
}

verify_checksum()
{
    if [[ $DISTRO == "centos" ]]; then
        _verify_checksum_centos
    else
        _verify_checksum_ubuntu
    fi
}

download_packages()
{
    mkdir -p $SCRIPT_DIR/neo
    cd $SCRIPT_DIR/neo
    
    if [[ $DISTRO == "centos" ]]; then
        _download_packages_centos
    else
        _download_packages_ubuntu
    fi
    verify_checksum
    if [[ $? -ne 0 ]]; then
        echo "ERROR: checksum do not match for downloaded packages"
        echo "       Please verify your Internet connection and make sure you have enough disk space or fix the problem manually and try again "
        exit $EXIT_FAILURE
    fi
}


version_gt() {
    # check if first version is greater than second version
    test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1";
}

summary()
{
    kernel_version=$(uname -r)

    echo
    echo Installation completed successfully.
    echo
    echo Next steps:
    echo "Add OpenCL users to the video group: 'sudo usermod -a -G video USERNAME'"
    echo "   e.g. if the user running OpenCL host applications is foo, run: sudo usermod -a -G video foo"
    echo "   Current user has been already added to the video group"
    echo

    # ask to install kernel 4.14 if current kernel version < 4.13 (GPU NEO driver supports only kernels 4.13.x and higher)
    if version_gt "4.13" "$kernel_version" ; then
        echo "Install 4.14 kernel using install_4_14_kernel.sh script and reboot into this kernel"
        echo
    fi

    echo "If you use 8th Generation Intel® Core™ processor, you will need to add:"
    echo "   i915.alpha_support=1"
    echo "   to the 4.14 kernel command line, in order to enable OpenCL functionality for this platform."
    echo
 
}

check_root_access()
{
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: you must run this script as root." >&2
        echo "Please try again with "sudo -E $0", or as root." >&2
        exit $EXIT_FAILURE
    fi
}

add_user_to_video_group()
{
    local real_user=$(logname 2>/dev/null || echo ${SUDO_USER:-${USER}})
    echo
    echo Adding $real_user to the video group...
    usermod -a -G video $real_user
    if [[ $? -ne 0 ]]; then
        echo WARNING: unable to add $real_user to the video group >&2
    fi
}

_check_distro_version()
{
    if [[ $DISTRO == centos ]]; then
        CENTOS_MINOR=$(sed 's/CentOS Linux release 7\.\([[:digit:]]\+\).\+/\1/' /etc/centos-release)
        if [[ $? -ne 0 ]]; then
            echo ERROR: failed to obtain CentOS version minor. >&2
            echo This script is supported only on CentOS 7 and above. >&2
            exit $EXIT_FAILURE
        fi
    elif [[ $DISTRO == ubuntu ]]; then
        UBUNTU_VERSION=$(lsb_release -r -s) 
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
    elif [[ -f /etc/lsb-release ]]; then
        DISTRO="ubuntu"
    fi

    _check_distro_version
}

check_agreement()
{
    if [ "$params" == "-y" ]; then
        return 0
    fi

    echo "This script will download and install Intel(R) Graphics Compute Runtime $INSTALL_DRIVER_VERSION, "
    echo "that was used to validate this OpenVINO package."
    echo "In case if you already have the driver - script will try to remove it."
    while true; do
        read -p "Want to proceed? (y/n): " yn
        case $yn in
            [Yy]*) return 0  ;;
            [Nn]*) exit 1 ;;
        esac
    done
}

check_current_driver()
{   
    echo "Checking current driver version..."
    if [[ $DISTRO == centos ]]; then
        gfx_version=$(yum info intel-opencl | grep Version)
    elif [[ $DISTRO == ubuntu ]]; then
        gfx_version=$(apt show intel-opencl | grep Version)
    fi
    gfx_version="$(echo -e "${gfx_version}" | sed -e 's/^Version\:[[:space:]]*//')"
    # install NEO OCL driver if current driver version < 19.41.14441
    if [[ ! -z $gfx_version && "$(printf '%s\n' "$INSTALL_DRIVER_VERSION" "$gfx_version" | sort -V | head -n 1)" = "$INSTALL_DRIVER_VERSION" ]]; then
        echo "Intel(R) Graphics Compute Runtime installation skipped because current version greater or equal to $INSTALL_DRIVER_VERSION" >&2
        echo "Installation of Intel Compute Runtime interrupted" >&2
        exit $EXIT_FAILURE
    else
        echo "Starting installation"
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
    echo "Intel OpenCL graphics driver installer"
    distro_init
    check_root_access
    check_current_driver
    check_agreement
    install
    summary
}

[[ "$0" == "$BASH_SOURCE" ]] && main "$@"
