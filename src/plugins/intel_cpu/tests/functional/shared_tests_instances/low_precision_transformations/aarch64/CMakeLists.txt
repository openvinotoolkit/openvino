# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(aarch64_SRC
    fully_connected_transformation.cpp
    mat_mul_transformation.cpp
)

# Convert to full paths
foreach(src_file ${aarch64_SRC})
    list(APPEND aarch64_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${aarch64_SRC_FULL})
