# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/insert_fake_quantize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/insert_fake_quantize.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/insert_requantize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/insert_requantize.hpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")