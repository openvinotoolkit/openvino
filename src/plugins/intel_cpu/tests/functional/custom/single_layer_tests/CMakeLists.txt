# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(single_layer_tests_SRC
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
    group_convolution.cpp
    group_convolution_backprop_data.cpp
    gru_cell.cpp
    gru_sequence.cpp
    log_softmax.cpp
    logical.cpp
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
    prior_box.cpp
    prior_box_clustered.cpp
    proposal.cpp
    psroi_pooling.cpp
    rdft.cpp
    region_yolo.cpp
    reorg_yolo.cpp
    reverse_sequence.cpp
    rnn_cell.cpp
    rnn_sequence.cpp
    roi_pooling.cpp
    roialign.cpp
    roll.cpp
    scatter_ND_update.cpp
    scatter_elements_update.cpp
    scatter_update.cpp
    select.cpp
    shape_ops.cpp
    shapeof.cpp
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

# Convert to full paths
foreach(src_file ${single_layer_tests_SRC})
    list(APPEND single_layer_tests_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${single_layer_tests_SRC_FULL})

add_subdirectory(classes)
add_subdirectory(instances)
