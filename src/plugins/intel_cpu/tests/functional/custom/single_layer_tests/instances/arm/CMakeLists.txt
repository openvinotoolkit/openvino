# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(arm_SRC
    conversion.cpp
    convolution_backprop_data.cpp
    eltwise.cpp
    interpolate.cpp
    matmul.cpp
    pooling.cpp
    reduce.cpp
)

# Convert to full paths
foreach(src_file ${arm_SRC})
    list(APPEND arm_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${arm_SRC_FULL})