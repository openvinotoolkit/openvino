# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME pytorch
                LINKABLE_FRONTEND
                SHUTDOWN_PROTOBUF
                FILEDESCRIPTION "FrontEnd to load and convert TorchScript models from PyTorch"
                LINK_LIBRARIES openvino::util openvino::core::dev)
