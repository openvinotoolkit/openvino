# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_sdp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_group_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_group_conv1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_reduce_multi_axis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deconv_multiple_output_edges.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_chain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_eltwise_convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reorder_deconv_nhwc.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")