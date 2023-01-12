# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME pytorch
                LINKABLE_FRONTEND
                #SHUTDOWN_PROTOBUF   # TODO: uncomment after solving missing google/protobuf headers
                FILEDESCRIPTION "FrontEnd to load and convert TorchScript models from Pytorch"
                LINK_LIBRARIES openvino::util openvino::runtime::dev)