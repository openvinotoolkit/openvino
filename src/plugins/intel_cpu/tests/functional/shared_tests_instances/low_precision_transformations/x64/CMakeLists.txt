# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(x64_SRC
    add_transformation.cpp
    assign_and_read_value_transformation.cpp
    batch_to_space_transformation.cpp
    clamp_transformation.cpp
    concat_transformation.cpp
    concat_with_child_and_output.cpp
    concat_with_different_precision_on_children.cpp
    concat_with_intermediate_transformation.cpp
    concat_with_neighbors_transformation.cpp
    concat_with_split_transformation.cpp
    convolution_backprop_data_transformation.cpp
    convolution_qdq_transformation.cpp
    convolution_transformation.cpp
    depth_to_space_transformation.cpp
    elementwise_branch_selection_transformation.cpp
    eliminate_fake_quantize_transformation.cpp
    fq_and_avg_pool_transformation.cpp
    fq_and_max_pool_transformation.cpp
    fq_and_two_output_branches_with_convolution.cpp
    fq_precision_selection_transformation.cpp
    fq_transformation.cpp
    fq_with_dq_not_optimal_transformation.cpp
    fully_connected_transformation.cpp
    fuse_convert_transformation.cpp
    fuse_dequantize_to_fq_transformation.cpp
    fuse_fq_and_scale_shift_transformation.cpp
    fuse_multiply_to_fq_transformation.cpp
    fuse_subtract_to_fq_transformation.cpp
    gather_transformation.cpp
    group_convolution_transformation.cpp
    groupconvolution_qdq_transformation.cpp
    interpolate_transformation.cpp
    mat_mul_transformation.cpp
    mat_mul_with_constant_transformation.cpp
    mat_mul_with_optimized_constant_fq.cpp
    move_fake_quantize_transformation.cpp
    multiply_to_group_convolution.cpp
    multiply_transformation.cpp
    multiply_with_one_parent.cpp
    mvn_transformation.cpp
    normalize_transformation.cpp
    output_layers.cpp
    pad_transformation.cpp
    prelu_transformation.cpp
    pull_reshape_through_dequantization.cpp
    recurrent_cell_transformation.cpp
    reduce_max_transformation.cpp
    reduce_mean_transformation.cpp
    reduce_min_transformation.cpp
    reduce_sum_transformation.cpp
    relu_transformation.cpp
    reshape_transformation.cpp
    shuffle_channels_transformation.cpp
    slice_transformation.cpp
    space_to_batch_transformation.cpp
    split_transformation.cpp
    squeeze_transformation.cpp
    strided_slice_transformation.cpp
    subtract_multiply_to_multiply_add.cpp
    subtract_transformation.cpp
    transpose_after_matmul_transformation.cpp
    transpose_transformation.cpp
    unsqueeze_transformation.cpp
    variadic_split_transformation.cpp
)

# Convert to full paths
foreach(src_file ${x64_SRC})
    list(APPEND x64_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${x64_SRC_FULL})
