# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Recurse to immediate children
add_subdirectory(classes)
add_subdirectory(instances)

# Non-ARM/AARCH64: list the top-level .cpp files that were previously appended at this directory level
if(NOT (ARM OR AARCH64))
  set(current_dir_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/augru_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/augru_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/batch_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/broadcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bucketize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/convert_to_plugin_specific_node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ctc_greedy_decoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ctc_greedy_decoder_seq_len.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ctc_loss.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cum_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op_internal_dyn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deformable_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/depth_to_space.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detection_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_bag_offsets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_bag_offsets_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_bag_packed.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_bag_packed_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_segments_sum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extract_image_patches.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eye.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_elements.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_nd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grid_sample.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/grn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_convolution_backprop_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/group_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gru_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gru_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lrn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lstm_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lstm_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_sparse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/non_max_suppression.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nonzero.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/normalize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_hot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prior_box_clustered.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/prior_box.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/proposal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/psroi_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rdft.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/region_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reorg_yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reverse_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn_cell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn_sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roialign.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roi_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/roll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_elements_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_ND_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scatter_update.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shapeof.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shape_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shuffle_channels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/slice_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/space_to_batch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/space_to_depth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/split.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/strided_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor_iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/topk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/unique.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/variadic_split.cpp
  )
  get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
  list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
  set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")
endif()