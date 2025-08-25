# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  concat_transformation.cpp
  concat_with_different_precision_on_children.cpp
  concat_with_neighbors_transformation.cpp
  concat_with_split_transformation.cpp
  fq_transformation.cpp
  fuse_convert_transformation.cpp
  fuse_dequantize_to_fq_transformation.cpp
  fully_connected_transformation.cpp
  groupconvolution_qdq_transformation.cpp
  mat_mul_with_optimized_constant_fq.cpp
  mvn_transformation.cpp
  output_layers.cpp
  recurrent_cell_transformation.cpp
  reduce_min_transformation.cpp
  reshape_transformation.cpp
  slice_transformation.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")