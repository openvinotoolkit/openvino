#!/bin/bash

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "Updating udev rules..."

if [ -z "$INTEL_OPENVINO_DIR" ]; then
    echo "Please set up your environment. Run 'source <OPENVINO_INSTALLDIR>/bin/setupvars.sh'."
    exit -1
fi

if [ -f "$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/97-myriad-usbboot.rules" ]; then
    sudo usermod -a -G users "$(whoami)"

    sudo cp "$INTEL_OPENVINO_DIR/deployment_tools/inference_engine/external/97-myriad-usbboot.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    sudo ldconfig
    echo "Udev rules have been successfully installed."
else
    echo "File '97-myriad-usbboot.rules' is missing. Please make sure you installed 'Inference Engine Runtime for Intel® Movidius™ VPU'."
    exit -1
fi 


