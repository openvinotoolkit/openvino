# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Files that live directly in this directory
set(current_dir_sources
  disable_gathercompressed_quantized_model.cpp
  quantized_matmuls_with_shared_weights.cpp
  subgraph_select_pd.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

get_property(subgraph_src_top_sources GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_SRC_TOP_SOURCES)
list(APPEND subgraph_src_top_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_SRC_TOP_SOURCES "${subgraph_src_top_sources}")

# Recurse to child directories
add_subdirectory(common)
add_subdirectory(classes)

if(X86 OR X86_64)
  add_subdirectory(x64)
endif()

if(ARM OR AARCH64)
  add_subdirectory(arm)
endif()