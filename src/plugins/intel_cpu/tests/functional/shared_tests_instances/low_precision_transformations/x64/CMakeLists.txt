# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_different_precision_on_children.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_neighbors_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_split_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fq_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_convert_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_dequantize_to_fq_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/groupconvolution_qdq_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_with_optimized_constant_fq.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mvn_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/output_layers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/recurrent_cell_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_min_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_transformation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/slice_transformation.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")