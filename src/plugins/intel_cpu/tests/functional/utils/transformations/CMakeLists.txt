# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  insert_fake_quantize.cpp
  insert_fake_quantize.hpp
  insert_requantize.cpp
  insert_requantize.hpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

# Add transformation .cpp files to required objects for per-test targets
set(required_transformation_sources
  insert_fake_quantize.cpp
  insert_requantize.cpp
)
list(TRANSFORM required_transformation_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(required_objects GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS)
list(APPEND required_objects ${required_transformation_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS "${required_objects}")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")