# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Files directly under shared_tests_instances/
set(current_dir_sources
  core_config.cpp
  skip_tests_config.cpp
  set_device_name.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Add these files to required objects for per-test targets
get_property(required_objects GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS)
list(APPEND required_objects ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS "${required_objects}")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

# Recurse to immediate children (each child lists only its own files)
add_subdirectory(execution_graph_tests)
add_subdirectory(preprocessing)
add_subdirectory(single_layer_tests)
add_subdirectory(snippets)
add_subdirectory(subgraph_tests)

# Arch-specific subtree
add_subdirectory(low_precision_transformations)

# ONNX subtree (guarded)
if(ENABLE_OV_ONNX_FRONTEND)
  add_subdirectory(onnx)
endif()