#!/bin/bash

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


