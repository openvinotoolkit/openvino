# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(quantized_SRC
    convolution.cpp
)

# Convert to full paths
foreach(src_file ${quantized_SRC})
    list(APPEND quantized_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${quantized_SRC_FULL})
