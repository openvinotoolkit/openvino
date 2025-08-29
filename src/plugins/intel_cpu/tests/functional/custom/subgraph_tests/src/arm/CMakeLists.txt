# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  concat_sdp.cpp
  convert_group_conv.cpp
  convert_group_conv1d.cpp
  convert_reduce_multi_axis.cpp
  deconv_multiple_output_edges.cpp
  eltwise_chain.cpp
  fuse_eltwise_convert.cpp
  matmul_weights_decompression.cpp
  reorder_deconv_nhwc.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

get_property(subgraph_arm_sources GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_ARM_SOURCES)
list(APPEND subgraph_arm_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_ARM_SOURCES "${subgraph_arm_sources}")
