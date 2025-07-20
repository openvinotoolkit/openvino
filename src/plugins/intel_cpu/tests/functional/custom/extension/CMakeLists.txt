# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(extension_SRC
    extension.cpp
)

# Convert to full paths
foreach(src_file ${extension_SRC})
    list(APPEND extension_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${extension_SRC_FULL})
