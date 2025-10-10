# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  concat_sdp.cpp
  concat_sdp.hpp
  conv_concat.cpp
  conv_concat.hpp
  conv_maxpool_activ.cpp
  conv_maxpool_activ.hpp
  eltwise_chain.cpp
  fuse_transpose_reorder.cpp
  fuse_transpose_reorder.hpp
  matmul_weights_decompression.cpp
  matmul_weights_decompression.hpp
  undefined_et.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Create a separate list of only .cpp files for class-based targets
set(class_cpp_files ${current_dir_sources})
list(FILTER class_cpp_files INCLUDE REGEX ".*\\.cpp$")
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_CLASS_SOURCES "${class_cpp_files}")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")