# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_sdp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_sdp.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_concat.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_maxpool_activ.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_maxpool_activ.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_chain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_transpose_reorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fuse_transpose_reorder.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_weights_decompression.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/undefined_et.cpp
)

# Create a separate list of only .cpp files for class-based targets
set(class_cpp_files ${current_dir_sources})
list(FILTER class_cpp_files INCLUDE REGEX ".*\\.cpp$")
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_CLASS_SOURCES "${class_cpp_files}")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")