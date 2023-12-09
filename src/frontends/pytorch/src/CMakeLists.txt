# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME pytorch
                FILEDESCRIPTION "FrontEnd to load and convert TorchScript, TorchFX models from PyTorch"
                LINK_LIBRARIES openvino::util openvino::core::dev)
