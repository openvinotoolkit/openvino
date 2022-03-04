#!/bin/bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "Updating udev rules..."

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    echo "Please set up your environment. Run 'source <OPENVINO_INSTALLDIR>/setupvars.sh'."
    exit -1
fi

if [ -f "$INTEL_OPENVINO_DIR/runtime/3rdparty/97-myriad-usbboot.rules" ]; then
    sudo usermod -a -G users "$(whoami)"

    sudo cp "$INTEL_OPENVINO_DIR/runtime/3rdparty/97-myriad-usbboot.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo ldconfig
    echo "Udev rules have been successfully installed."
else
    echo "File '97-myriad-usbboot.rules' is missing. Please make sure you installed 'Inference Engine Runtime for Intel® Movidius™ VPU'."
    exit -1
fi 
