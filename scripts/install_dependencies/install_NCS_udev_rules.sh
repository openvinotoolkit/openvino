#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" >/dev/null 2>&1 && pwd )"

echo "Updating udev rules..."

if [ -f "$SCRIPT_DIR/97-myriad-usbboot.rules" ]; then
    sudo usermod -a -G users "$(whoami)"

    sudo cp "$SCRIPT_DIR/97-myriad-usbboot.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo ldconfig
    echo "Udev rules have been successfully installed."
else
    echo "File '97-myriad-usbboot.rules' is missing. Please make sure you installed 'Inference Engine Runtime for Intel® Movidius™ VPU'."
    exit -1
fi 
