# Primitive Test Mapping and Bench Rules

This document maps bench primitives to GPU plugin unit/fusion tests and records
which test patterns are reflected in bench code.

## Scope

- Source benches: src/plugins/intel_gpu/tests/bench_kernel/primitives/bench_*.cpp
- Unit tests: src/plugins/intel_gpu/tests/unit/test_cases/*
- Fusion tests: src/plugins/intel_gpu/tests/unit/fusions/*

## Rules Reflected in Bench Code

1. For primitives that parse post-ops chains, bench topology should model chain order used by fusions tests.
2. Perf mode should keep terminal post-op node non-output by attaching a temporary consumer reorder.
3. Accuracy mode should apply the same post-op chain in CPU reference using apply_post_ops_ref.
4. If first post-op token is consumed as primitive main op (for example eltwise/activation), only remaining tokens are chained.

## Mapping Snapshot

The following mapping was generated from filename pattern matching and reviewed manually for post-op-heavy primitives.

```
bench_adaptive_pooling.cpp      UNIT:-  FUSION:-
bench_arg_max_min.cpp   UNIT:arg_max_gpu_test.cpp       FUSION:-
bench_batch_to_space.cpp        UNIT:batch_to_space_gpu_test.cpp        FUSION:batch_to_space_fusion_test.cpp
bench_border.cpp        UNIT:border_gpu_test.cpp        FUSION:-
bench_broadcast.cpp     UNIT:broadcast_gpu_test.cpp     FUSION:broadcast_fusion_test.cpp
bench_bucketize.cpp     UNIT:bucketize_gpu_test.cpp     FUSION:-
bench_col2im.cpp        UNIT:-  FUSION:-
bench_conv.cpp  UNIT:convolution_gpu_test.cpp   FUSION:convolution_fusion_test.cpp
bench_convert_color.cpp UNIT:convert_color_gpu_test.cpp FUSION:-
bench_crop.cpp  UNIT:crop_gpu_test.cpp  FUSION:-
bench_ctc_greedy_decoder.cpp    UNIT:-  FUSION:-
bench_ctc_loss.cpp      UNIT:ctc_loss_gpu_test.cpp      FUSION:-
bench_cum_sum.cpp       UNIT:cum_sum_gpu_test.cpp       FUSION:-
bench_deconvolution.cpp UNIT:deconvolution_gpu_test.cpp FUSION:deconvolution_fusion_test.cpp
bench_depth_to_space.cpp        UNIT:depth_to_space_gpu_test.cpp        FUSION:depth_to_space_fusion_test.cpp
bench_detection_output.cpp      UNIT:detection_output_test.cpp;experimental_detectron_detection_output_gpu_test.cpp     FUSION:-
bench_dft.cpp   UNIT:dft_gpu_test.cpp   FUSION:-
bench_dynamic_quantize.cpp      UNIT:dynamic_quantize_gpu_test.cpp      FUSION:-
bench_eltwise.cpp       UNIT:eltwise_gpu_test.cpp       FUSION:eltwise_fusion_test.cpp
bench_embedding_bag.cpp UNIT:embedding_bag_gpu_test.cpp FUSION:-
bench_exp_detectron_detection_output.cpp        UNIT:-  FUSION:-
bench_exp_detectron_gen_proposals.cpp   UNIT:-  FUSION:-
bench_exp_detectron_prior_grid.cpp      UNIT:-  FUSION:-
bench_exp_detectron_roi_feature.cpp     UNIT:-  FUSION:-
bench_exp_detectron_topk_rois.cpp       UNIT:-  FUSION:-
bench_extract_image_patches.cpp UNIT:extract_image_patches_gpu_test.cpp FUSION:-
bench_eye.cpp   UNIT:-  FUSION:-
bench_fake_convert.cpp  UNIT:-  FUSION:-
bench_fc.cpp    UNIT:fully_connected_gpu_test.cpp       FUSION:fully_connected_fusion_test.cpp
bench_gather.cpp        UNIT:gather_elements_gpu_test.cpp;gather_gpu_test.cpp;gather_nd_gpu_test.cpp;gather_tree_gpu_test.cpp;moe_gather_gpu_test.cpp   FUSION:gather_elements_fusion_test.cpp;gather_fusion_test.cpp;gather_nd_fusion_test.cpp
bench_gather_elements.cpp       UNIT:gather_elements_gpu_test.cpp       FUSION:gather_elements_fusion_test.cpp
bench_gather_nd.cpp     UNIT:gather_nd_gpu_test.cpp     FUSION:gather_nd_fusion_test.cpp
bench_gather_tree.cpp   UNIT:gather_tree_gpu_test.cpp   FUSION:-
bench_gemm.cpp  UNIT:gemm_gpu_test.cpp;moe_gemm_gpu_test.cpp    FUSION:gemm_fusion_test.cpp
bench_generate_proposals.cpp    UNIT:experimental_detectron_generate_proposals_single_image_gpu_test.cpp;generate_proposals_gpu_test.cpp        FUSION:-
bench_grid_sample.cpp   UNIT:grid_sample_gpu_test.cpp   FUSION:-
bench_grn.cpp   UNIT:-  FUSION:-
bench_group_normalization.cpp   UNIT:group_normalization_gpu_test.cpp   FUSION:-
bench_istft.cpp UNIT:istft_gpu_test.cpp FUSION:-
bench_lrn.cpp   UNIT:lrn_gpu_test.cpp   FUSION:lrn_fusion_test.cpp
bench_lstm_cell.cpp     UNIT:-  FUSION:-
bench_matrix_nms.cpp    UNIT:matrix_nms_gpu_test.cpp    FUSION:-
bench_misc.cpp  UNIT:-  FUSION:-
bench_multiclass_nms.cpp        UNIT:multiclass_nms_gpu_test.cpp        FUSION:-
bench_multinomial.cpp   UNIT:-  FUSION:-
bench_mvn.cpp   UNIT:mvn_gpu_test.cpp   FUSION:mvn_fusion_test.cpp
bench_non_max_suppression.cpp   UNIT:non_max_suppression_test.cpp       FUSION:-
bench_non_zero.cpp      UNIT:non_zero_gpu_test.cpp      FUSION:-
bench_normalize.cpp     UNIT:-  FUSION:normalize_fusion_test.cpp
bench_one_hot.cpp       UNIT:one_hot_gpu_test.cpp       FUSION:-
bench_pooling.cpp       UNIT:adaptive_avg_pooling_gpu_test.cpp;adaptive_max_pooling_gpu_test.cpp;pooling_gpu_test.cpp;roi_pooling_gpu_test.cpp  FUSION:pooling_fusion_test.cpp
bench_prior_box.cpp     UNIT:prior_box_gpu_test.cpp     FUSION:-
bench_proposal.cpp      UNIT:proposal_cpu_test.cpp;proposal_test_data.cpp      FUSION:-
bench_quantize.cpp      UNIT:dynamic_quantize_gpu_test.cpp;quantize_gpu_test.cpp       FUSION:-
bench_random_uniform.cpp        UNIT:random_uniform_gpu_test.cpp        FUSION:-
bench_range.cpp UNIT:range_gpu_test.cpp FUSION:-
bench_reduce.cpp        UNIT:reduce_gpu_test.cpp        FUSION:reduce_fusion_test.cpp
bench_region_yolo.cpp   UNIT:region_yolo_gpu_test.cpp   FUSION:-
bench_reorg_yolo.cpp    UNIT:reorg_yolo_gpu_test.cpp    FUSION:-
bench_resample.cpp      UNIT:resample_gpu_test.cpp      FUSION:resample_fusion_test.cpp
bench_reverse.cpp       UNIT:reverse_gpu_test.cpp;reverse_sequence_gpu_test.cpp FUSION:-
bench_reverse_sequence.cpp      UNIT:reverse_sequence_gpu_test.cpp      FUSION:-
bench_rms.cpp   UNIT:rms_gpu_test.cpp   FUSION:rms_fusion_test.cpp
bench_roi_align.cpp     UNIT:roi_align_gpu_test.cpp;roi_align_rotated_gpu_test.cpp      FUSION:-
bench_roi_pooling.cpp   UNIT:roi_pooling_gpu_test.cpp   FUSION:-
bench_roll.cpp  UNIT:roll_gpu_test.cpp  FUSION:-
bench_rope.cpp  UNIT:-  FUSION:-
bench_scatter_elements_update.cpp       UNIT:scatter_elements_update_gpu_test.cpp       FUSION:scatter_elements_update_fusion_test.cpp
bench_scatter_nd_update.cpp     UNIT:scatter_nd_update_gpu_test.cpp     FUSION:scatter_nd_update_fusion_test.cpp
bench_scatter_update.cpp        UNIT:scatter_update_gpu_test.cpp        FUSION:scatter_update_fusion_test.cpp
bench_sdpa.cpp  UNIT:sdpa_gpu_test.cpp  FUSION:-
bench_search_sorted.cpp UNIT:search_sorted_gpu_test.cpp FUSION:-
bench_segment_max.cpp   UNIT:segment_max_gpu_test.cpp   FUSION:-
bench_select.cpp        UNIT:select_gpu_test.cpp        FUSION:select_fusion_test.cpp
bench_shuffle_channels.cpp      UNIT:shuffle_channels_test.cpp  FUSION:-
bench_slice.cpp UNIT:slice_gpu_test.cpp;strided_slice_gpu_test.cpp      FUSION:strided_slice_fusion_test.cpp
bench_softmax.cpp       UNIT:softmax_gpu_test.cpp       FUSION:softmax_fusion_test.cpp
bench_space_to_batch.cpp        UNIT:space_to_batch_gpu_test.cpp        FUSION:space_to_batch_fusion_test.cpp
bench_space_to_depth.cpp        UNIT:space_to_depth_gpu_test.cpp        FUSION:space_to_depth_fusion_test.cpp
bench_sparse_fill_empty_rows.cpp        UNIT:sparse_fill_empty_rows_gpu_test.cpp        FUSION:-
bench_stft.cpp  UNIT:stft_gpu_benchamrk.cpp;stft_gpu_test.cpp   FUSION:-
bench_strided_slice.cpp UNIT:strided_slice_gpu_test.cpp FUSION:strided_slice_fusion_test.cpp
bench_swiglu.cpp        UNIT:swiglu_gpu_test.cpp        FUSION:-
bench_tile.cpp  UNIT:tile_gpu_test.cpp  FUSION:-
bench_unique.cpp        UNIT:unique_gpu_test.cpp        FUSION:-
```

