# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(common_SRC
    activation.cpp
    bitwise_shift.cpp
    col2im.cpp
    conversion.cpp
    convolution.cpp
    convolution_backprop_data.cpp
    eltwise.cpp
    extremum.cpp
    interpolate.cpp
    inverse.cpp
    matmul.cpp
    multinomial.cpp
    mvn.cpp
    nms_rotated.cpp
    pooling.cpp
    random_uniform.cpp
    reduce.cpp
    scaled_attn.cpp
    segment_max.cpp
    softmax.cpp
    sparse_fill_empty_rows.cpp
    string_tensor_pack.cpp
    string_tensor_unpack.cpp
    transpose.cpp
)

# Convert to full paths
foreach(src_file ${common_SRC})
    list(APPEND common_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${common_SRC_FULL})