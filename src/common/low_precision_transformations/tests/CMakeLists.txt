# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_lp_transformations_tests)

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/add_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/align_concat_quantization_parameters_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/assign_and_read_value_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/avg_pool_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/avg_pool_with_child_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_to_space_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/clamp_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compose_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_selection_with_intermediate_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_different_precision_on_children.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_fq_tranformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_intermediate_precision_selection_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_intermediate_reshape_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_intermediate_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_intermediate_with_constant_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_neighbors_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_neighbors_transformation_with_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_not_quantized_parent_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_reshape_at_the_end_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_split_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat_with_strided_slice_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_subtract_constant_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution_backprop_data_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution_qdq_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convolution_with_incorrect_weights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/depth_to_space_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_with_multi_parent_dequantization_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eliminate_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_transformation_is_broadcasted_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_and_two_output_branches_with_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_on_weights_with_unsupported_child.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_precision_selection_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_with_dq_not_optimal_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_with_dynamic_intervals_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fold_convert_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fold_fake_quantize_in_transformations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fuse_convert_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fuse_dequantize_to_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fuse_fake_quantize_with_multi_inputs_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fuse_multiply_to_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fuse_subtract_to_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/get_dequantization_below_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/get_dequantization_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/get_dequantization_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_convolution_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interpolate_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/is_asymmetric_on_weights_dequantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/is_asymmetric_on_weights_fq.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/is_constant_path_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/is_function_quantized_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache_static_quantization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/layer_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lpt_avoid_shapeof_propagation_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mark_dequantization_subgraph_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/markup_avg_pool_precisions_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/markup_bias_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_with_constant_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/max_pool_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/move_dequantization_after_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/move_fake_quantize_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multiply_partial_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multiply_to_group_convolution_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multiply_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mvn_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/normalize_dequantization_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/normalize_l2_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pad_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/precision_details_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prelu_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pull_reshape_through_dequantization_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pull_transpose_through_dequantization_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/quantization_granularity_restriction_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/recurrent_cell_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_max_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_mean_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_min_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_sum_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/relu_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reshape_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/round_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/separate_in_standalone_branch_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shuffle_channels_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_low_precision_transformer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/space_to_batch_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/split_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/squeeze_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/strided_slice_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph/src/fq_decomposition_with_shared_constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph/src/qdq_stripping.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transformations_after_split_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transformer_is_function_quantized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/transpose_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/calclulate_levels_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/data_precision_check.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/layer_transformation_get_data_precision.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/reshape_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/shared_attribute_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/update_reshape_values.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unsqueeze_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/variadic_split_transformation.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/layer_transformation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reduce_transformation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simple_low_precision_transformer.hpp
)

source_group("src"     FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_executable(${TARGET_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${TARGET_NAME} PRIVATE
    gtest
    gtest_main
    openvino::runtime::dev
    common_test_utils
    ov_lpt_models
    gmock
)

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_property(TEST ${TARGET_NAME} PROPERTY LABELS OV UNIT LP_TRANSFORMATIONS)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)

ov_build_target_faster(${TARGET_NAME} PCH)
