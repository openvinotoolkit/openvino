# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Files that live directly in utils/
set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_test_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_test_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fusing_test_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fusing_test_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/filter_cpu_info.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantization_utils.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/properties_test.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution_params.hpp
)

# Arch-specific files that live in utils/
if(X86_64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/arm/filter_cpu_info.cpp)
elseif(RISCV64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/riscv64/filter_cpu_info.cpp)
endif()

# Append to global
get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

# Immediate child directory
add_subdirectory(transformations)