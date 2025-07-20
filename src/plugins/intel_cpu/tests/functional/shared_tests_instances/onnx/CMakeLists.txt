# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(onnx_SRC
    quantized_models_tests.cpp
)

# Convert to full paths
foreach(src_file ${onnx_SRC})
    list(APPEND onnx_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${onnx_SRC_FULL})
