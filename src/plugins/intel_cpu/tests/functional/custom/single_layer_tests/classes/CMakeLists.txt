# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if(ARM OR AARCH64)
  # ARM/AARCH64: list only the explicitly enabled class files from the original ARM block
  set(current_dir_sources
    activation.cpp
    activation.hpp
    bitwise_shift.cpp
    bitwise_shift.hpp
    col2im.cpp
    col2im.hpp
    comparison.cpp
    comparison.hpp
    conversion.cpp
    conversion.hpp
    convolution_backprop_data.cpp
    convolution_backprop_data.hpp
    convolution.cpp
    convolution.hpp
    eltwise.cpp
    eltwise.hpp
    extremum.cpp
    extremum.hpp
    interpolate.cpp
    interpolate.hpp
    logical.cpp
    logical.hpp
    matmul.cpp
    matmul.hpp
    mvn.cpp
    mvn.hpp
    pooling.cpp
    pooling.hpp
    random_uniform.cpp
    random_uniform.hpp
    reduce.cpp
    reduce.hpp
    rms_norm.cpp
    rms_norm.hpp
    scaled_attn.cpp
    scaled_attn.hpp
    segment_max.cpp
    segment_max.hpp
    softmax.cpp
    softmax.hpp
    sparse_fill_empty_rows.cpp
    sparse_fill_empty_rows.hpp
    string_tensor_pack.cpp
    string_tensor_pack.hpp
    string_tensor_unpack.cpp
    string_tensor_unpack.hpp
    transpose.cpp
    transpose.hpp
  )
  list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
else()
  # Non-ARM/AARCH64: list the full set of class files
  set(current_dir_sources
    activation.cpp
    activation.hpp
    bitwise_shift.cpp
    bitwise_shift.hpp
    col2im.cpp
    col2im.hpp
    comparison.cpp
    comparison.hpp
    conversion.cpp
    conversion.hpp
    convolution_backprop_data.cpp
    convolution_backprop_data.hpp
    convolution.cpp
    convolution.hpp
    eltwise.cpp
    eltwise.hpp
    extremum.cpp
    extremum.hpp
    interpolate.cpp
    interpolate.hpp
    logical.cpp
    logical.hpp
    matmul.cpp
    matmul.hpp
    mvn.cpp
    mvn.hpp
    pooling.cpp
    pooling.hpp
    random_uniform.cpp
    random_uniform.hpp
    reduce.cpp
    reduce.hpp
    rms_norm.cpp
    rms_norm.hpp
    scaled_attn.cpp
    scaled_attn.hpp
    segment_max.cpp
    segment_max.hpp
    softmax.cpp
    softmax.hpp
    sparse_fill_empty_rows.cpp
    sparse_fill_empty_rows.hpp
    string_tensor_pack.cpp
    string_tensor_pack.hpp
    string_tensor_unpack.cpp
    string_tensor_unpack.hpp
    transpose.cpp
    transpose.hpp
  )
  list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
endif()

# Create a separate list of only .cpp files for class-based targets
set(class_cpp_files ${current_dir_sources})
list(FILTER class_cpp_files INCLUDE REGEX ".*\\.cpp$")
set_property(GLOBAL PROPERTY OV_CPU_FUNC_SLT_CLASS_SOURCES "${class_cpp_files}")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")