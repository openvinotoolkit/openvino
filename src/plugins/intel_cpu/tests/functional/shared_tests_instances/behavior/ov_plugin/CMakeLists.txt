# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(ov_plugin_SRC
    caching_tests.cpp
    core_integration.cpp
    core_threading_tests.cpp
    life_time.cpp
    plugin_name.cpp
    properties_tests.cpp
    version.cpp
)

# Convert to full paths
foreach(src_file ${ov_plugin_SRC})
    list(APPEND ov_plugin_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${ov_plugin_SRC_FULL})
