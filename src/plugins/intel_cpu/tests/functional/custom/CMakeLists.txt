# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Files directly under custom/ (behavior + guarded extension)
set(current_dir_sources
  behavior/export_import.cpp
  behavior/ov_executable_network/concurent_release_memory.cpp
  behavior/ov_executable_network/properties.cpp
  behavior/ov_executable_network/release_memory.cpp
  behavior/ov_plugin/properties.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

if(ENABLE_OV_ONNX_FRONTEND)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/extension/extension.cpp)
endif()

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

# Recurse to immediate children
add_subdirectory(single_layer_tests)
add_subdirectory(subgraph_tests)