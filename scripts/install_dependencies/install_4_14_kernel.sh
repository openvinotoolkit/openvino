#!/bin/bash -x

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script installs Linux kernel 4.14 required for Intel NEO OpenCL driver on Ubuntu and CentOS

if grep -i "rhel" /etc/os-release &>/dev/null; then
	# Cent OS
	echo "install kernel build dependencies"
	sudo -E yum install -y git gcc gcc-c++ ncurses-devel openssl-devel bc xz elfutils-libelf-devel xorg-x11-drv-nouveau rpm-build

	echo "download 4.14.20 kernel"
	if [ ! -f ./linux-4.14.20.tar.xz ]; then
		wget https://www.kernel.org/pub/linux/kernel/v4.x/linux-4.14.20.tar.xz
	fi

	tar -xJf linux-4.14.20.tar.xz
	cd linux-4.14.20
	echo "build 4.14.20 kernel"
	make olddefconfig

	make -j 8 binrpm-pkg
	cd ~/rpmbuild/RPMS/x86_64
	sudo -E yum -y localinstall *.rpm
	sudo -E grub2-set-default 0

elif grep -i "ubuntu" /etc/os-release &>/dev/null; then
	# Ubuntu
	sudo -E add-apt-repository ppa:teejee2008/ppa
	sudo -E apt-get update && sudo apt-get install -y ukuu
	sudo -E ukuu --install v4.14.20
fi