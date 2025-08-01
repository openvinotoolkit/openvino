# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(behavior)
add_subdirectory(single_layer_tests)
add_subdirectory(subgraph_tests)

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(extension)
endif()