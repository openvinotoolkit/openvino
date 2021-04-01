#!/bin/bash -x

# Copyright (c) 2018-2019 Intel Corporation
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