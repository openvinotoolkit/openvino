# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(classes_SRC
    activation.cpp
    activation.hpp
    bitwise_shift.cpp
    bitwise_shift.hpp
    col2im.cpp
    col2im.hpp
    conversion.cpp
    conversion.hpp
    convolution.cpp
    convolution.hpp
    convolution_backprop_data.cpp
    convolution_backprop_data.hpp
    eltwise.cpp
    eltwise.hpp
    extremum.cpp
    extremum.hpp
    interpolate.cpp
    interpolate.hpp
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

# Convert to full paths
foreach(src_file ${classes_SRC})
    list(APPEND classes_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${classes_SRC_FULL})