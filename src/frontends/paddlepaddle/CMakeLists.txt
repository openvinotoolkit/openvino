# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: Add LINKABLE_FRONTEND option when tensorflow frontend directory is moved to openvino folder
ov_add_frontend(NAME paddlepaddle
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format"
                LINK_LIBRARIES openvino::runtime::dev)
