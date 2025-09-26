# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Files that live directly in utils/
set(current_dir_sources
  cpu_test_utils.cpp
  cpu_test_utils.hpp
  fusing_test_utils.cpp
  fusing_test_utils.hpp
  filter_cpu_info.hpp
  quantization_utils.hpp
  properties_test.hpp
  convolution_params.hpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Arch-specific files that live in utils/
if(X86_64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/arm/filter_cpu_info.cpp)
elseif(RISCV64)
  list(APPEND current_dir_sources ${CMAKE_CURRENT_SOURCE_DIR}/riscv64/filter_cpu_info.cpp)
endif()

# Add specific utils files to required objects for per-test targets
set(required_utils_sources
  cpu_test_utils.cpp
  fusing_test_utils.cpp
)
list(TRANSFORM required_utils_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Add arch-specific filter_cpu_info.cpp
if(X86_64)
  list(APPEND required_utils_sources ${CMAKE_CURRENT_SOURCE_DIR}/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
  list(APPEND required_utils_sources ${CMAKE_CURRENT_SOURCE_DIR}/arm/filter_cpu_info.cpp)
elseif(RISCV64)
  list(APPEND required_utils_sources ${CMAKE_CURRENT_SOURCE_DIR}/riscv64/filter_cpu_info.cpp)
endif()

get_property(required_objects GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS)
list(APPEND required_objects ${required_utils_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS "${required_objects}")

# Append to global
get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

# Immediate child directory
add_subdirectory(transformations)