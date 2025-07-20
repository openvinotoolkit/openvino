# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(shl_SRC
    matmul.cpp
)

# Convert to full paths
foreach(src_file ${shl_SRC})
    list(APPEND shl_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${shl_SRC_FULL})
