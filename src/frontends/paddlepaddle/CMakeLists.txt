# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: Add LINKABLE_FRONTEND option when tensorflow frontend directory is moved to openvino folder
ov_add_frontend(NAME paddlepaddle
                PROTOBUF_LITE
                FILEDESCRIPTION "FrontEnd to load and convert PaddlePaddle file format"
                LINK_LIBRARIES inference_engine_transformations openvino::util)

# Add include path to so_extension.hpp
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/frontend.cpp
        PROPERTIES INCLUDE_DIRECTORIES "${OpenVINO_SOURCE_DIR}/src/core/src/")