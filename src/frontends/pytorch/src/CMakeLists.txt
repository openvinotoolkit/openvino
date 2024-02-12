# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(${CMAKE_CXX_STANDARD} GREATER_EQUAL 20)
    ov_deprecated_featurs_no_errors()
endif()

ov_add_frontend(NAME pytorch
                LINKABLE_FRONTEND
                SHUTDOWN_PROTOBUF
                FILEDESCRIPTION "FrontEnd to load and convert TorchScript models from PyTorch"
                LINK_LIBRARIES openvino::util openvino::core::dev)
