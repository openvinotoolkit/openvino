# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(arm_SRC
    concat_sdp.cpp
    convert_group_conv.cpp
    convert_group_conv1d.cpp
    convert_reduce_multi_axis.cpp
    deconv_multiple_output_edges.cpp
    eltwise_chain.cpp
    fuse_eltwise_convert.cpp
    matmul_weights_decompression.cpp
    reorder_deconv_nhwc.cpp
)

# Convert to full paths
foreach(src_file ${arm_SRC})
    list(APPEND arm_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${arm_SRC_FULL})
