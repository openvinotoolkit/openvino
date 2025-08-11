# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


function(collect_utils_sources out_list)
  set(local_srcs)
  set(utils_dir ${CMAKE_CURRENT_SOURCE_DIR}/utils)
  
  # Utils (common)
  list(APPEND local_srcs
    ${utils_dir}/cpu_test_utils.cpp
    ${utils_dir}/cpu_test_utils.hpp
    ${utils_dir}/fusing_test_utils.cpp
    ${utils_dir}/fusing_test_utils.hpp
    ${utils_dir}/filter_cpu_info.hpp
    ${utils_dir}/quantization_utils.hpp
    ${utils_dir}/properties_test.hpp
    ${utils_dir}/convolution_params.hpp
    ${utils_dir}/transformations/insert_fake_quantize.cpp
    ${utils_dir}/transformations/insert_fake_quantize.hpp
    ${utils_dir}/transformations/insert_requantize.cpp
    ${utils_dir}/transformations/insert_requantize.hpp)

  # Utils (arch-specific)
  if(X86_64)
    list(APPEND local_srcs ${utils_dir}/x64/filter_cpu_info.cpp)
  elseif(ARM OR AARCH64)
    list(APPEND local_srcs ${utils_dir}/arm/filter_cpu_info.cpp)
  elseif(RISCV64)
    list(APPEND local_srcs ${utils_dir}/riscv64/filter_cpu_info.cpp)
  endif()

  # Return the collected sources
  set(${out_list} "${local_srcs}" PARENT_SCOPE)
endfunction()