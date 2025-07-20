# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(behavior_SRC
    export_import.cpp
)

# Convert to full paths
foreach(src_file ${behavior_SRC})
    list(APPEND behavior_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${behavior_SRC_FULL})

add_subdirectory(ov_executable_network)
add_subdirectory(ov_plugin)
