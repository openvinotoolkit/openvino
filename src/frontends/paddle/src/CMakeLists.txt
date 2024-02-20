# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ov_add_frontend(NAME paddle
                LINKABLE_FRONTEND
                PROTOBUF_REQUIRED
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format"
                LINK_LIBRARIES openvino::util openvino::core::dev)
