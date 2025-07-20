# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(x64_SRC
    activation.cpp
    col2im.cpp
    conversion.cpp
    convolution.cpp
    convolution_backprop_data.cpp
    eltwise.cpp
    interpolate.cpp
    matmul.cpp
    mvn.cpp
    pooling.cpp
    random_uniform.cpp
    reduce.cpp
    rms_norm.cpp
    softmax.cpp
    transpose.cpp
)

# Convert to full paths
foreach(src_file ${x64_SRC})
    list(APPEND x64_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${x64_SRC_FULL})

if(ENABLE_MLAS_FOR_CPU)
    add_subdirectory(mlas)
endif()

add_subdirectory(quantized)
