# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Recurse to immediate children
add_subdirectory(classes)
add_subdirectory(instances)

# Non-ARM/AARCH64: list the top-level .cpp files that were previously appended at this directory level
if(NOT (ARM OR AARCH64))
  set(current_dir_sources
    adaptive_pooling.cpp
    augru_cell.cpp
    augru_sequence.cpp
    batch_to_space.cpp
    broadcast.cpp
    bucketize.cpp
    concat.cpp
    convert_to_plugin_specific_node.cpp
    ctc_greedy_decoder.cpp
    ctc_greedy_decoder_seq_len.cpp
    ctc_loss.cpp
    cum_sum.cpp
    custom_op_internal_dyn.cpp
    deformable_convolution.cpp
    depth_to_space.cpp
    detection_output.cpp
    embedding_bag_offsets.cpp
    embedding_bag_offsets_sum.cpp
    embedding_bag_packed.cpp
    embedding_bag_packed_sum.cpp
    embedding_segments_sum.cpp
    extract_image_patches.cpp
    eye.cpp
    fake_quantize.cpp
    gather.cpp
    gather_elements.cpp
    gather_nd.cpp
    gather_tree.cpp
    grid_sample.cpp
    grn.cpp
    group_convolution_backprop_data.cpp
    group_convolution.cpp
    gru_cell.cpp
    gru_sequence.cpp
    log_softmax.cpp
    loop.cpp
    lrn.cpp
    lstm_cell.cpp
    lstm_sequence.cpp
    matmul_sparse.cpp
    non_max_suppression.cpp
    nonzero.cpp
    normalize.cpp
    one_hot.cpp
    pad.cpp
    prior_box_clustered.cpp
    prior_box.cpp
    proposal.cpp
    psroi_pooling.cpp
    rdft.cpp
    region_yolo.cpp
    reorg_yolo.cpp
    reverse_sequence.cpp
    rnn_cell.cpp
    rnn_sequence.cpp
    roialign.cpp
    roi_pooling.cpp
    roll.cpp
    scatter_elements_update.cpp
    scatter_ND_update.cpp
    scatter_update.cpp
    select.cpp
    shapeof.cpp
    shape_ops.cpp
    shuffle_channels.cpp
    slice.cpp
    slice_scatter.cpp
    space_to_batch.cpp
    space_to_depth.cpp
    split.cpp
    strided_slice.cpp
    tensor_iterator.cpp
    tile.cpp
    topk.cpp
    unique.cpp
    variadic_split.cpp
  )
  list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
  get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
  list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
  set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

  get_property(slt_sources GLOBAL PROPERTY OV_CPU_FUNC_SLT_SOURCES)
  list(APPEND slt_sources ${current_dir_sources})
  set_property(GLOBAL PROPERTY OV_CPU_FUNC_SLT_SOURCES "${slt_sources}")
endif()