# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(x64_SRC
    filter_cpu_info.cpp
)

# Convert to full paths
foreach(src_file ${x64_SRC})
    list(APPEND x64_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${x64_SRC_FULL})
