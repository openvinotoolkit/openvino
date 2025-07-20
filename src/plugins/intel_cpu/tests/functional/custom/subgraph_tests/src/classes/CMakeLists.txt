# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(classes_SRC
    concat_sdp.cpp
    concat_sdp.hpp
    conv_concat.cpp
    conv_concat.hpp
    conv_maxpool_activ.cpp
    conv_maxpool_activ.hpp
    eltwise_chain.cpp
    eltwise_chain.hpp
    fuse_transpose_reorder.cpp
    fuse_transpose_reorder.hpp
    matmul_weights_decompression.cpp
    matmul_weights_decompression.hpp
    undefined_et.cpp
)

# Convert to full paths
foreach(src_file ${classes_SRC})
    list(APPEND classes_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${classes_SRC_FULL})
