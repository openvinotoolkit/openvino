# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_shape_inference")

set(SHAPE_INFER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Create static library
add_library(${TARGET_NAME} STATIC)

target_sources(${TARGET_NAME}
    PRIVATE

    # Sources
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution_shape_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/experimental_detectron_shape_infer_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_data_accessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp

    # Headers
    ${SHAPE_INFER_INCLUDE_DIR}/adaptive_avg_pool_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/adaptive_max_pool_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/assign_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/augru_cell_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/augru_sequence_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/avg_pool_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/batch_norm_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/batch_to_space_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/binary_convolution_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/broadcast_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/bucketize_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/col2im_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/compare.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/concat_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/convolution_backprop_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/convolution_backprop_shape_inference_util.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/convolution_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/convolution_shape_inference_util.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/copy_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/ctc_greedy_decoder_seq_len_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/ctc_greedy_decoder_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/ctc_loss_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/deformable_convolution_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/deformable_psroi_pooling_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/depth_to_space_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/detection_output_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/dimension_util.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/einsum_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/element_visitor.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/eltwise_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/embedding_segments_sum_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/embedding_shape_infer_utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/embeddingbag_offsets_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/embeddingbag_packed_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_detection_output_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_generate_proposals_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_prior_grid_generator_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_roi_feature_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_shape_infer_utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/experimental_detectron_topkrois_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/extract_image_patches_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/eye_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/fake_convert_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/fake_quantize.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/fft_base_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/fft_common_validation.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gated_delta_net_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gather_elements_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gather_nd_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gather_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gather_tree_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/generate_proposals_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/glu_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/grid_sample_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/group_convolution_backprop_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/group_convolution_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/group_normalization_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/grouped_matmul_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gru_cell_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/gru_sequence_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/i420_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/identity_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/interpolate_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/inverse_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/irdft_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/istft_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/lstm_cell_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/lstm_sequence_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/matmul_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/matrix_nms_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/max_pool_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/multiclass_nms_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/multinomial_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/nms_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/nv12_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/one_hot_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/pa_kv_reorder_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/pad_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/paged_attention_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/paged_causal_conv1d_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/paged_gated_delta_net_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/pooling_shape_inference_util.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/prior_box_clustered_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/prior_box_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/prior_box_shape_inference_util.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/proposal_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/psroi_pooling_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/random_uniform_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/range_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/rdft_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/reduce_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/region_yolo_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/reorg_yolo_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/reshape_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/reverse_sequence_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/reverse_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/rms_norm_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/rnn_base_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/rnn_cell_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/rnn_sequence_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/roi_align_rotated_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/roi_align_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/roi_align_shape_utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/roi_pooling_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/roll_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/scaled_dot_product_attention_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/scatter_elements_update_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/scatter_nd_base_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/search_sorted_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/segment_max_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/select_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/sequence_generator.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/shape_infer_type_utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/shape_nodes.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/shape_validation.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/shuffle_channels_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/slice_scatter_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/slice_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/slice_shape_inference_utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/space_to_batch_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/space_to_depth_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/sparse_fill_empty_rows_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/split_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/squeeze_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/stft_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/strided_slice_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/string_tensor_pack_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/string_tensor_unpack_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/tensor_data_accessor.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/tile_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/topk_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/transpose_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/unsqueeze_shape_inference.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/utils.hpp
    ${SHAPE_INFER_INCLUDE_DIR}/variadic_split_shape_inference.hpp
)

add_library(openvino::shape_inference ALIAS ${TARGET_NAME})
set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME shape_inference)

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${SHAPE_INFER_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>
    $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

# developer package

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

ov_developer_package_export_targets(TARGET ${TARGET_NAME}
                                    INSTALL_INCLUDE_DIRECTORIES "${SHAPE_INFER_INCLUDE_DIR}/")
