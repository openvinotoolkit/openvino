// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// bench_kernel - A kernel-level GPU benchmarking tool for OpenVINO GPU plugin
//
// Usage:
//   ./ov_gpu_bench_kernel --fc --dt=f16 1x4096:4096x4096
//   ./ov_gpu_bench_kernel --gemm --dt=f16 64x128:128x256
//   ./ov_gpu_bench_kernel --fc --dt=f16:i4:f16 --attr-scales=wei:per_ocic:f16:1x128 1x4096:4096x4096
//   ./ov_gpu_bench_kernel --list-devices
//   ./ov_gpu_bench_kernel --list-kernels

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <memory>
#include <map>

#include <gflags/gflags.h>

#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/device_query.hpp>

#include "common/bench_config.hpp"
#include "common/bench_types.hpp"
#include "common/bench_timer.hpp"
#include "primitives/kernel_registry.hpp"

// ============================================================================
// gflags definitions
// ============================================================================

// Device
DEFINE_int32(device, 0, "GPU device index (default: 0)");
DEFINE_bool(list_devices, false, "List available GPU devices and exit");
DEFINE_bool(list_kernels, false, "List available kernel benchmarks and exit");

// Kernel selection (only one should be set)
DEFINE_bool(fc, false, "Run fully_connected kernel benchmark");
DEFINE_bool(gemm, false, "Run gemm kernel benchmark");
DEFINE_bool(conv, false, "Run convolution kernel benchmark");
DEFINE_bool(sdpa, false, "Run scaled_dot_product_attention kernel benchmark");
DEFINE_bool(eltwise, false, "Run eltwise kernel benchmark");
DEFINE_bool(softmax, false, "Run softmax kernel benchmark");
DEFINE_bool(reduce, false, "Run reduce kernel benchmark");
DEFINE_bool(activation, false, "Run activation kernel benchmark");
DEFINE_bool(pooling, false, "Run pooling kernel benchmark");
DEFINE_bool(mvn, false, "Run mvn kernel benchmark");
DEFINE_bool(reorder, false, "Run reorder kernel benchmark");
DEFINE_bool(permute, false, "Run permute kernel benchmark");
DEFINE_bool(concatenation, false, "Run concatenation kernel benchmark");
DEFINE_bool(deconv, false, "Run deconvolution kernel benchmark");
DEFINE_bool(resample, false, "Run resample kernel benchmark");
DEFINE_bool(rms, false, "Run rms kernel benchmark");
DEFINE_bool(swiglu_standalone, false, "Run swiglu kernel benchmark");
DEFINE_bool(dynamic_quantize, false, "Run dynamic_quantize kernel benchmark");
DEFINE_bool(gather, false, "Run gather kernel benchmark");
DEFINE_bool(crop, false, "Run crop kernel benchmark");
DEFINE_bool(rope, false, "Run rope kernel benchmark");
DEFINE_bool(strided_slice, false, "Run strided_slice kernel benchmark");
DEFINE_bool(broadcast, false, "Run broadcast kernel benchmark");
DEFINE_bool(select, false, "Run select kernel benchmark");
DEFINE_bool(scatter_update, false, "Run scatter_update kernel benchmark");
DEFINE_bool(tile, false, "Run tile kernel benchmark");
DEFINE_bool(normalize, false, "Run normalize kernel benchmark");
DEFINE_bool(gather_elements, false, "Run gather_elements kernel benchmark");
DEFINE_bool(scatter_nd_update, false, "Run scatter_nd_update kernel benchmark");
DEFINE_bool(scatter_elements_update, false, "Run scatter_elements_update kernel benchmark");
DEFINE_bool(group_normalization, false, "Run group_normalization kernel benchmark");
DEFINE_bool(quantize, false, "Run quantize kernel benchmark");
DEFINE_bool(adaptive_pooling, false, "Run adaptive_pooling kernel benchmark");
DEFINE_bool(arg_max_min, false, "Run arg_max_min kernel benchmark");
DEFINE_bool(col2im, false, "Run col2im kernel benchmark");
DEFINE_bool(detection_output, false, "Run detection_output kernel benchmark");
DEFINE_bool(cum_sum, false, "Run cum_sum kernel benchmark");
DEFINE_bool(depth_to_space, false, "Run depth_to_space kernel benchmark");
DEFINE_bool(space_to_depth, false, "Run space_to_depth kernel benchmark");
DEFINE_bool(grn, false, "Run grn kernel benchmark");
DEFINE_bool(lrn, false, "Run lrn kernel benchmark");
DEFINE_bool(reorg_yolo, false, "Run reorg_yolo kernel benchmark");
DEFINE_bool(roll, false, "Run roll kernel benchmark");
DEFINE_bool(shuffle_channels, false, "Run shuffle_channels kernel benchmark");
DEFINE_bool(non_zero, false, "Run non_zero kernel benchmark");
DEFINE_bool(bucketize, false, "Run bucketize kernel benchmark");
DEFINE_bool(gather_nd, false, "Run gather_nd kernel benchmark");
DEFINE_bool(gather_tree, false, "Run gather_tree kernel benchmark");
DEFINE_bool(one_hot, false, "Run one_hot kernel benchmark");
DEFINE_bool(reverse, false, "Run reverse kernel benchmark");
DEFINE_bool(reverse_sequence, false, "Run reverse_sequence kernel benchmark");
DEFINE_bool(search_sorted, false, "Run search_sorted kernel benchmark");
DEFINE_bool(slice, false, "Run slice kernel benchmark");
DEFINE_bool(border, false, "Run border kernel benchmark");
DEFINE_bool(batch_to_space, false, "Run batch_to_space kernel benchmark");
DEFINE_bool(space_to_batch, false, "Run space_to_batch kernel benchmark");
DEFINE_bool(grid_sample, false, "Run grid_sample kernel benchmark");
DEFINE_bool(roi_align, false, "Run roi_align kernel benchmark");
DEFINE_bool(roi_pooling, false, "Run roi_pooling kernel benchmark");
DEFINE_bool(extract_image_patches, false, "Run extract_image_patches kernel benchmark");
DEFINE_bool(convert_color, false, "Run convert_color kernel benchmark");
DEFINE_bool(region_yolo, false, "Run region_yolo kernel benchmark");
DEFINE_bool(dft, false, "Run dft kernel benchmark");
DEFINE_bool(range, false, "Run range kernel benchmark");
DEFINE_bool(random_uniform, false, "Run random_uniform kernel benchmark");
DEFINE_bool(eye, false, "Run eye kernel benchmark");
DEFINE_bool(embedding_bag, false, "Run embedding_bag kernel benchmark");
DEFINE_bool(fake_convert, false, "Run fake_convert kernel benchmark");
DEFINE_bool(ctc_greedy_decoder, false, "Run ctc_greedy_decoder kernel benchmark");
DEFINE_bool(non_max_suppression, false, "Run non_max_suppression kernel benchmark");
DEFINE_bool(prior_box, false, "Run prior_box kernel benchmark");
DEFINE_bool(proposal, false, "Run proposal kernel benchmark");
DEFINE_bool(ctc_loss, false, "Run ctc_loss kernel benchmark");
DEFINE_bool(stft, false, "Run STFT kernel benchmark");
DEFINE_bool(istft, false, "Run ISTFT kernel benchmark");
DEFINE_bool(lstm_cell, false, "Run lstm_cell kernel benchmark");
DEFINE_bool(matrix_nms, false, "Run matrix_nms kernel benchmark");
DEFINE_bool(multiclass_nms, false, "Run multiclass_nms kernel benchmark");
DEFINE_bool(multinomial, false, "Run multinomial kernel benchmark");
DEFINE_bool(segment_max, false, "Run segment_max kernel benchmark");
DEFINE_bool(sparse_fill_empty_rows, false, "Run sparse_fill_empty_rows kernel benchmark");
DEFINE_bool(unique_count, false, "Run unique_count kernel benchmark");
DEFINE_bool(unique_gather, false, "Run unique_gather kernel benchmark");
DEFINE_bool(generate_proposals, false, "Run generate_proposals kernel benchmark");
DEFINE_bool(exp_detectron_detection_output, false, "Run experimental_detectron_detection_output kernel benchmark");
DEFINE_bool(exp_detectron_generate_proposals, false, "Run experimental_detectron_generate_proposals_single_image kernel benchmark");
DEFINE_bool(exp_detectron_prior_grid_gen, false, "Run experimental_detectron_prior_grid_generator kernel benchmark");
DEFINE_bool(exp_detectron_roi_feature, false, "Run experimental_detectron_roi_feature_extractor kernel benchmark");
DEFINE_bool(exp_detectron_topk_rois, false, "Run experimental_detectron_topk_rois kernel benchmark");

// Data types & impl
DEFINE_string(dt, "f16", "Data type(s), colon-separated (e.g., f16, f16:i4:f16)");
DEFINE_string(impl, "", "Force implementation type: ocl, onednn, or empty for auto");
DEFINE_string(force_impl, "", "Force specific OCL kernel by name (e.g., gemm_tiled_opt). Requires --impl=ocl");

// Layout formats (from BenchVerbose log)
DEFINE_string(in_layouts, "", "Input layout formats, comma-separated (e.g., bfyx,b_fs_yx_fsv16)");
DEFINE_string(out_layouts, "", "Output layout formats, comma-separated (e.g., b_fs_yx_fsv16)");

// Attributes
DEFINE_string(attr_scales, "", "Scale attributes (e.g., wei:per_ocic:f16:1x128)");
DEFINE_string(attr_zero_points, "", "Zero point attributes (e.g., wei:per_ocic:i4:1x128)");
DEFINE_string(attr_post_ops, "", "Post-ops (e.g., relu, sum:f16:per_oc+relu)");

// Shapes (alternative to positional argument)
DEFINE_string(shapes, "", "Shapes string (e.g., 1x4096:4096x4096). Can also be passed as positional arg.");

// Execution control
DEFINE_int32(warmup, 5, "Number of warmup iterations");
DEFINE_int32(iters, 20, "Number of measurement iterations");
DEFINE_double(max_ms, 3000.0, "Max milliseconds per problem");
DEFINE_int32(verbose, 1, "Verbosity level (0=silent, 1=summary, 2=detail)");
DEFINE_string(mode, "p", "Benchmark mode: c(correctness), p(perf), f(fast perf), cp(corr+perf), r(run 1iter)");

// Reorder-specific
DEFINE_bool(truncate, false, "Reorder truncation mode (Convert op uses truncate=true)");

// Primitive-specific attributes (from verbose log)
DEFINE_int32(transpose_a, 0, "Gemm: transpose input0 (0|1)");
DEFINE_int32(transpose_b, 0, "Gemm: transpose input1 (0|1)");
DEFINE_string(gemm_order0, "", "Gemm: input0 transpose order (e.g. 0:1:2:3)");
DEFINE_string(gemm_order1, "", "Gemm: input1 transpose order (e.g. 0:1:2:3)");
DEFINE_string(gemm_order_out, "", "Gemm: output transpose order (e.g. 0:1:2:3)");
DEFINE_int32(is_causal, -1, "SDPA: causal mask (-1=auto, 0=false, 1=true)");
DEFINE_string(order_q, "", "SDPA: Q transpose order (e.g. 0:2:1:3)");
DEFINE_string(order_k, "", "SDPA: K transpose order");
DEFINE_string(order_v, "", "SDPA: V transpose order");
DEFINE_string(order_out, "", "SDPA: output transpose order");
DEFINE_string(scale_val, "", "SDPA: explicit scale value");
DEFINE_int32(compressed, -1, "FullyConnected: compressed weights hint (-1=auto, 0|1)");
DEFINE_int32(dynamic_quantized, -1, "FullyConnected: dynamic quantized activation hint (-1=auto, 0|1)");
DEFINE_int32(dynamic_quantized_zp, -1, "FullyConnected: dynamic quantized activation zp hint (-1=auto, 0|1)");
DEFINE_int32(dynamic_quantized_precomputed_reduction, -1, "FullyConnected: dynamic quantized precomputed reduction hint (-1=auto, 0|1)");
DEFINE_int32(fc_input_size, -1, "FullyConnected: input rank hint (-1=auto)");
DEFINE_int32(fc_weights_rank, -1, "FullyConnected: weights rank hint (-1=auto)");
DEFINE_int32(groups, 1, "Conv: filter groups");
DEFINE_string(strides, "", "Conv: stride (e.g. 2x2)");
DEFINE_string(dilations, "", "Conv: dilation (e.g. 1x1)");
DEFINE_string(padding_begin, "", "Conv: padding begin (e.g. 1x1)");
DEFINE_string(padding_end, "", "Conv: padding end (e.g. 1x1)");
DEFINE_int32(grouped_weights_shape, 0, "Conv: explicit group dim in weights (0|1)");
DEFINE_int32(pool_mode, 0, "Pooling: mode (0=max, 1=avg, 2=avg_no_pad)");
DEFINE_string(kernel, "", "Pooling: kernel size (e.g. 2x2)");
DEFINE_string(pool_strides, "", "Pooling: stride (e.g. 2x2)");
DEFINE_string(pads_begin, "", "Pooling: padding begin");
DEFINE_string(pads_end, "", "Pooling: padding end");
DEFINE_int32(rounding_type, 0, "Pooling: rounding (0=floor, 1=ceil)");
DEFINE_int32(reduce_mode, 1, "Reduce: mode (1=mean)");
DEFINE_int32(keep_dims, 1, "Reduce: keep dims (0|1)");
DEFINE_string(reduce_axes, "", "Reduce: axes (e.g. 1:2:3)");
DEFINE_int32(axis, -1, "Softmax/Gather: axis (-1=last dim)");
DEFINE_int32(normalize_variance, 1, "MVN: normalize variance (0|1)");
DEFINE_string(epsilon, "", "MVN: epsilon value");
DEFINE_int32(eps_inside_sqrt, 0, "MVN: epsilon inside sqrt (0|1)");
DEFINE_string(mvn_reduction_axes, "", "MVN: reduction axes (e.g., 1:2:3)");
DEFINE_int32(eltwise_mode, -1, "Eltwise: mode (-1=from post-ops)");
DEFINE_int32(pythondiv, 0, "Eltwise: python-style division (0|1)");
DEFINE_string(eltwise_coefficients, "", "Eltwise: coefficients for sum mode (e.g. 1.0:0.5)");
DEFINE_string(eltwise_stride, "", "Eltwise: per-input stride tensors (semicolon separated)");
DEFINE_string(eltwise_broadcast_type, "", "Eltwise: auto broadcast type (numpy|pdpd|none)");
DEFINE_int32(eltwise_broadcast_axis, -1, "Eltwise: auto broadcast axis for PDPD");
DEFINE_int32(glu_type, 0, "SwiGLU: GLU type (0=Swish)");
DEFINE_int32(split_axis, -1, "SwiGLU: split axis (-1=last)");
DEFINE_int32(split_length, -1, "SwiGLU: split length (-1=auto)");
DEFINE_int32(gate_idx, 0, "SwiGLU: gate index (0|1)");
DEFINE_int32(gather_axis, 0, "Gather: axis");
DEFINE_int32(batch_dim, 0, "Gather: batch dimension");
DEFINE_int32(support_neg_ind, 0, "Gather: support negative indices (0|1)");
DEFINE_int32(head_cnt, 0, "RoPE: head count");
DEFINE_int32(head_size, 0, "RoPE: head size");
DEFINE_int32(rotary_ndims, 0, "RoPE: rotary dimensions");
DEFINE_int32(is_interleaved, 0, "RoPE: interleaved mode (0|1)");
DEFINE_int32(is_chatglm, 0, "RoPE: ChatGLM mode (0|1)");
DEFINE_int32(is_qwen, 0, "RoPE: Qwen mode (0|1)");
DEFINE_int32(input_trans0213, 1, "RoPE: input transpose 0213 (0|1)");
DEFINE_int32(slice_start, 0, "RoPE: slice start");
DEFINE_int32(slice_stop, 0, "RoPE: slice stop");
DEFINE_int32(gather_rank, 0, "RoPE: gather rank");
DEFINE_string(offsets, "", "Crop: offsets (e.g. 0:0:0:0)");
DEFINE_string(ss_begin, "", "StridedSlice: begin indices");
DEFINE_string(ss_end, "", "StridedSlice: end indices");
DEFINE_string(ss_strides, "", "StridedSlice: strides");
DEFINE_string(begin_mask, "", "StridedSlice: begin mask");
DEFINE_string(end_mask, "", "StridedSlice: end mask");
DEFINE_string(shrink_axis_mask, "", "StridedSlice: shrink axis mask");
DEFINE_string(new_axis_mask, "", "StridedSlice: new axis mask");
DEFINE_int32(concat_axis, 1, "Concatenation: axis");

// Tile-specific
DEFINE_string(tile_repeats, "", "Tile: repeats (e.g. 2:2:1:1)");

// Normalize-specific
DEFINE_int32(across_spatial, 0, "Normalize: across spatial (0|1)");

// GroupNormalization-specific
DEFINE_int32(num_groups, 1, "GroupNormalization: number of groups");

// Quantize-specific
DEFINE_int32(levels, 0, "Quantize: FakeQuantize levels (e.g. 256)");

// ScatterNDUpdate-specific
DEFINE_int32(indices_rank, 0, "ScatterNDUpdate: indices rank");

// Resample-specific
DEFINE_string(resample_sizes, "", "Resample: output sizes (e.g. 1:64:64:64)");
DEFINE_int32(resample_mode, 0, "Resample: mode (0=nearest, 1=linear, 2=cubic)");

// Permute-specific
DEFINE_string(permute_order, "", "Permute: order (e.g. 0:2:1:3)");

// Broadcast-specific
DEFINE_string(broadcast_axes, "", "Broadcast: axes (e.g. 0:1)");
DEFINE_string(broadcast_target, "", "Broadcast: target shape (e.g. 1:32:128:128)");

// AdaptivePooling-specific
DEFINE_int32(adaptive_pool_mode, 0, "AdaptivePooling: mode (0=avg, 1=max)");
DEFINE_string(adaptive_pool_out, "", "AdaptivePooling: output sizes (e.g. 1:64:7:7)");

// ArgMaxMin (TopK)-specific
DEFINE_int32(topk_mode, 0, "ArgMaxMin: mode (0=max, 1=min)");
DEFINE_int32(top_k, 1, "ArgMaxMin: number of top elements");

// Col2Im-specific
DEFINE_string(col2im_output_shape, "", "Col2Im: output spatial shape (e.g. 4x4)");
DEFINE_string(col2im_kernel_shape, "", "Col2Im: kernel shape (e.g. 3x3)");
DEFINE_string(col2im_padding_begin, "", "Col2Im: padding begin (e.g. 1x1)");
DEFINE_string(col2im_padding_end, "", "Col2Im: padding end (e.g. 1x1)");

// ScatterElementsUpdate-specific
DEFINE_int32(scatter_mode, 0, "ScatterElementsUpdate: reduction mode");
DEFINE_int32(scatter_use_init_val, 1, "ScatterElementsUpdate: use initial value (0|1)");

// DetectionOutput-specific
DEFINE_int32(det_num_classes, 21, "DetectionOutput: number of classes");
DEFINE_int32(det_keep_top_k, 200, "DetectionOutput: keep top k");
DEFINE_int32(det_top_k, -1, "DetectionOutput: top k for NMS");
DEFINE_double(det_nms_threshold, 0.45, "DetectionOutput: NMS threshold");
DEFINE_double(det_confidence_threshold, 0.01, "DetectionOutput: confidence threshold");
DEFINE_int32(det_code_type, 1, "DetectionOutput: code type (0=corner, 1=center_size, 2=corner_size)");
DEFINE_int32(det_share_location, 1, "DetectionOutput: share location (0|1)");
DEFINE_int32(det_background_label_id, 0, "DetectionOutput: background label id");
DEFINE_int32(det_variance_encoded, 0, "DetectionOutput: variance encoded in target (0|1)");
DEFINE_double(det_eta, 1.0, "DetectionOutput: eta (adaptive NMS)");
DEFINE_int32(det_prior_info_size, 4, "DetectionOutput: prior info size");
DEFINE_int32(det_prior_coordinates_offset, 0, "DetectionOutput: prior coordinates offset");
DEFINE_int32(det_prior_is_normalized, 1, "DetectionOutput: prior is normalized (0|1)");
DEFINE_int32(det_input_width, -1, "DetectionOutput: input width (-1=auto)");
DEFINE_int32(det_input_height, -1, "DetectionOutput: input height (-1=auto)");
DEFINE_int32(det_decrease_label_id, 0, "DetectionOutput: decrease label id (0|1)");
DEFINE_int32(det_clip_before_nms, 0, "DetectionOutput: clip before NMS (0|1)");
DEFINE_int32(det_clip_after_nms, 0, "DetectionOutput: clip after NMS (0|1)");
DEFINE_double(det_objectness_score, 0.0, "DetectionOutput: objectness score");

// New primitive-specific attributes
DEFINE_int32(cum_exclusive, 0, "CumSum: exclusive (0|1)");
DEFINE_int32(cum_reverse, 0, "CumSum: reverse (0|1)");
DEFINE_int32(block_size, 0, "DepthToSpace/SpaceToDepth/ReorgYolo: block size");
DEFINE_int32(d2s_mode, 0, "DepthToSpace/SpaceToDepth: mode (0=blocks_first, 1=depth_first)");
DEFINE_double(grn_bias, 1e-6, "GRN: bias");
DEFINE_int32(lrn_size, 5, "LRN: normalization size");
DEFINE_double(lrn_k, 1.0, "LRN: k parameter");
DEFINE_double(lrn_alpha, 0.0001, "LRN: alpha");
DEFINE_double(lrn_beta, 0.75, "LRN: beta");
DEFINE_int32(lrn_norm_region, 0, "LRN: norm region (0=across_ch, 1=within_ch)");
DEFINE_string(roll_shift, "", "Roll: shift values (e.g. 2:3:1)");
DEFINE_int32(shuffle_group, 2, "ShuffleChannels: group");
DEFINE_int32(border_mode, 0, "Border: pad mode (0=constant, 1=edge, 2=reflect, 3=symmetric)");
DEFINE_double(border_value, 0.0, "Border: pad value for constant mode");
DEFINE_int32(reverse_mode, 0, "Reverse: mode (0=index, 1=mask)");
DEFINE_int32(seq_axis, 1, "ReverseSequence: sequence axis");
DEFINE_int32(one_hot_depth, 0, "OneHot: depth");
DEFINE_double(one_hot_on_value, 1.0, "OneHot: on value");
DEFINE_double(one_hot_off_value, 0.0, "OneHot: off value");
DEFINE_int32(grid_mode, 0, "GridSample: mode (0=bilinear, 1=bicubic, 2=nearest)");
DEFINE_int32(grid_padding, 0, "GridSample: padding (0=zeros, 1=border, 2=reflection)");
DEFINE_int32(grid_align_corners, 0, "GridSample: align corners (0|1)");
DEFINE_int32(roi_pooled_h, 7, "ROI: pooled height");
DEFINE_int32(roi_pooled_w, 7, "ROI: pooled width");
DEFINE_double(roi_spatial_scale, 0.0625, "ROI: spatial scale");
DEFINE_int32(roi_sampling_ratio, 2, "ROI: sampling ratio");
DEFINE_string(eip_sizes, "", "ExtractImagePatches: patch sizes (e.g. 3x3)");
DEFINE_string(eip_strides, "", "ExtractImagePatches: strides (e.g. 1x1)");
DEFINE_string(eip_rates, "", "ExtractImagePatches: rates (e.g. 1x1)");
DEFINE_int32(yolo_coords, 4, "RegionYolo: coordinates");
DEFINE_int32(yolo_classes, 80, "RegionYolo: classes");
DEFINE_int32(yolo_num, 3, "RegionYolo: num anchors");
DEFINE_int32(yolo_do_softmax, 0, "RegionYolo: do softmax (0|1)");
DEFINE_int32(dft_inverse, 0, "DFT: inverse (0|1)");
DEFINE_double(range_start, 0.0, "Range: start value");
DEFINE_double(range_stop, 100.0, "Range: stop value");
DEFINE_double(range_step, 1.0, "Range: step value");
DEFINE_int32(eye_diagonal, 0, "Eye: diagonal offset");
DEFINE_int32(nms_top_k, 100, "NMS: top k");
DEFINE_double(nms_iou_threshold, 0.5, "NMS: IoU threshold");
DEFINE_double(nms_score_threshold, 0.0, "NMS: score threshold");

// Batch file
DEFINE_string(batch, "", "Batch file with problem definitions");

// ============================================================================
// Helper functions
// ============================================================================

static std::string detect_kernel_name() {
    // Map gflag bools to kernel names
    static const std::vector<std::pair<bool*, std::string>> kernel_flags = {
        {&FLAGS_fc,               "fully_connected"},
        {&FLAGS_gemm,             "gemm"},
        {&FLAGS_conv,             "convolution"},
        {&FLAGS_sdpa,             "scaled_dot_product_attention"},
        {&FLAGS_eltwise,          "eltwise"},
        {&FLAGS_softmax,          "softmax"},
        {&FLAGS_reduce,           "reduce"},
        {&FLAGS_activation,       "activation"},
        {&FLAGS_pooling,          "pooling"},
        {&FLAGS_mvn,              "mvn"},
        {&FLAGS_reorder,          "reorder"},
        {&FLAGS_permute,          "permute"},
        {&FLAGS_concatenation,    "concatenation"},
        {&FLAGS_deconv,           "deconvolution"},
        {&FLAGS_resample,         "resample"},
        {&FLAGS_rms,              "rms"},
        {&FLAGS_swiglu_standalone,"swiglu"},
        {&FLAGS_dynamic_quantize, "dynamic_quantize"},
        {&FLAGS_gather,           "gather"},
        {&FLAGS_crop,             "crop"},
        {&FLAGS_rope,             "rope"},
        {&FLAGS_strided_slice,    "strided_slice"},
        {&FLAGS_broadcast,        "broadcast"},
        {&FLAGS_select,           "select"},
        {&FLAGS_scatter_update,   "scatter_update"},
        {&FLAGS_tile,             "tile"},
        {&FLAGS_normalize,        "normalize"},
        {&FLAGS_gather_elements,  "gather_elements"},
        {&FLAGS_scatter_nd_update, "scatter_nd_update"},
        {&FLAGS_scatter_elements_update, "scatter_elements_update"},
        {&FLAGS_group_normalization, "group_normalization"},
        {&FLAGS_quantize,         "quantize"},
        {&FLAGS_adaptive_pooling, "adaptive_pooling"},
        {&FLAGS_arg_max_min,      "arg_max_min"},
        {&FLAGS_col2im,           "col2im"},
        {&FLAGS_detection_output, "detection_output"},
        {&FLAGS_cum_sum,          "cum_sum"},
        {&FLAGS_depth_to_space,   "depth_to_space"},
        {&FLAGS_space_to_depth,   "space_to_depth"},
        {&FLAGS_grn,              "grn"},
        {&FLAGS_lrn,              "lrn"},
        {&FLAGS_reorg_yolo,       "reorg_yolo"},
        {&FLAGS_roll,             "roll"},
        {&FLAGS_shuffle_channels, "shuffle_channels"},
        {&FLAGS_non_zero,         "non_zero"},
        {&FLAGS_bucketize,        "bucketize"},
        {&FLAGS_gather_nd,        "gather_nd"},
        {&FLAGS_gather_tree,      "gather_tree"},
        {&FLAGS_one_hot,          "one_hot"},
        {&FLAGS_reverse,          "reverse"},
        {&FLAGS_reverse_sequence, "reverse_sequence"},
        {&FLAGS_search_sorted,    "search_sorted"},
        {&FLAGS_slice,            "slice"},
        {&FLAGS_border,           "border"},
        {&FLAGS_batch_to_space,   "batch_to_space"},
        {&FLAGS_space_to_batch,   "space_to_batch"},
        {&FLAGS_grid_sample,      "grid_sample"},
        {&FLAGS_roi_align,        "roi_align"},
        {&FLAGS_roi_pooling,      "roi_pooling"},
        {&FLAGS_extract_image_patches, "extract_image_patches"},
        {&FLAGS_convert_color,    "convert_color"},
        {&FLAGS_region_yolo,      "region_yolo"},
        {&FLAGS_dft,              "dft"},
        {&FLAGS_range,            "range"},
        {&FLAGS_random_uniform,   "random_uniform"},
        {&FLAGS_eye,              "eye"},
        {&FLAGS_embedding_bag,    "embedding_bag"},
        {&FLAGS_fake_convert,     "fake_convert"},
        {&FLAGS_ctc_greedy_decoder, "ctc_greedy_decoder"},
        {&FLAGS_non_max_suppression, "non_max_suppression"},
        {&FLAGS_prior_box,        "prior_box"},
        {&FLAGS_proposal,         "proposal"},
        {&FLAGS_ctc_loss,         "ctc_loss"},
        {&FLAGS_stft,             "STFT"},
        {&FLAGS_istft,            "ISTFT"},
        {&FLAGS_lstm_cell,        "lstm_cell"},
        {&FLAGS_matrix_nms,       "matrix_nms"},
        {&FLAGS_multiclass_nms,   "multiclass_nms"},
        {&FLAGS_multinomial,      "multinomial"},
        {&FLAGS_segment_max,      "segment_max"},
        {&FLAGS_sparse_fill_empty_rows, "sparse_fill_empty_rows"},
        {&FLAGS_unique_count,     "unique_count"},
        {&FLAGS_unique_gather,    "unique_gather"},
        {&FLAGS_generate_proposals, "generate_proposals"},
        {&FLAGS_exp_detectron_detection_output, "experimental_detectron_detection_output"},
        {&FLAGS_exp_detectron_generate_proposals, "experimental_detectron_generate_proposals_single_image"},
        {&FLAGS_exp_detectron_prior_grid_gen, "experimental_detectron_prior_grid_generator"},
        {&FLAGS_exp_detectron_roi_feature, "experimental_detectron_roi_feature_extractor"},
        {&FLAGS_exp_detectron_topk_rois, "experimental_detectron_topk_rois"},
    };

    std::string found;
    for (const auto& kv : kernel_flags) {
        if (*kv.first) {
            if (!found.empty()) {
                std::cerr << "Error: Multiple kernel types specified. Use only one." << std::endl;
                return "";
            }
            found = kv.second;
        }
    }
    return found;
}

static bench_kernel::bench_config build_config(const std::string& kernel_name,
                                                const std::string& shapes_str) {
    bench_kernel::bench_config cfg;
    cfg.kernel_name          = kernel_name;
    cfg.device               = FLAGS_device;
    cfg.list_devices         = FLAGS_list_devices;
    cfg.warmup_iters         = FLAGS_warmup;
    cfg.perf_iters           = FLAGS_iters;
    cfg.max_ms_per_prb       = FLAGS_max_ms;
    cfg.verbose              = FLAGS_verbose;
    cfg.mode_str             = FLAGS_mode;
    cfg.dt_str               = FLAGS_dt;
    cfg.impl_str             = FLAGS_impl;
    cfg.force_impl_str       = FLAGS_force_impl;
    cfg.in_layouts_str       = FLAGS_in_layouts;
    cfg.out_layouts_str      = FLAGS_out_layouts;
    cfg.shapes_str           = shapes_str;
    cfg.attr_scales_str      = FLAGS_attr_scales;
    cfg.attr_zero_points_str = FLAGS_attr_zero_points;
    cfg.attr_post_ops_str    = FLAGS_attr_post_ops;
    cfg.truncate             = FLAGS_truncate;
    // Primitive-specific attributes
    cfg.transpose_a          = FLAGS_transpose_a;
    cfg.transpose_b          = FLAGS_transpose_b;
    cfg.gemm_order0          = FLAGS_gemm_order0;
    cfg.gemm_order1          = FLAGS_gemm_order1;
    cfg.gemm_order_out       = FLAGS_gemm_order_out;
    cfg.is_causal            = FLAGS_is_causal;
    cfg.order_q              = FLAGS_order_q;
    cfg.order_k              = FLAGS_order_k;
    cfg.order_v              = FLAGS_order_v;
    cfg.order_out            = FLAGS_order_out;
    cfg.scale_val            = FLAGS_scale_val;
    cfg.compressed           = FLAGS_compressed;
    cfg.dynamic_quantized    = FLAGS_dynamic_quantized;
    cfg.dynamic_quantized_zp = FLAGS_dynamic_quantized_zp;
    cfg.dynamic_quantized_precomputed_reduction = FLAGS_dynamic_quantized_precomputed_reduction;
    cfg.fc_input_size        = FLAGS_fc_input_size;
    cfg.fc_weights_rank      = FLAGS_fc_weights_rank;
    cfg.groups               = FLAGS_groups;
    cfg.strides              = FLAGS_strides;
    cfg.dilations            = FLAGS_dilations;
    cfg.padding_begin        = FLAGS_padding_begin;
    cfg.padding_end          = FLAGS_padding_end;
    cfg.grouped_weights_shape = FLAGS_grouped_weights_shape;
    cfg.pool_mode            = FLAGS_pool_mode;
    cfg.pool_kernel          = FLAGS_kernel;
    cfg.pool_strides         = FLAGS_pool_strides;
    cfg.pads_begin           = FLAGS_pads_begin;
    cfg.pads_end             = FLAGS_pads_end;
    cfg.rounding_type        = FLAGS_rounding_type;
    cfg.reduce_mode          = FLAGS_reduce_mode;
    cfg.keep_dims            = FLAGS_keep_dims;
    cfg.reduce_axes          = FLAGS_reduce_axes;
    cfg.axis                 = FLAGS_axis;
    cfg.normalize_variance   = FLAGS_normalize_variance;
    cfg.epsilon              = FLAGS_epsilon;
    cfg.eps_inside_sqrt      = FLAGS_eps_inside_sqrt;
    cfg.mvn_reduction_axes   = FLAGS_mvn_reduction_axes;
    cfg.eltwise_mode         = FLAGS_eltwise_mode;
    cfg.pythondiv            = FLAGS_pythondiv;
    cfg.eltwise_coefficients = FLAGS_eltwise_coefficients;
    cfg.eltwise_stride       = FLAGS_eltwise_stride;
    cfg.eltwise_broadcast_type = FLAGS_eltwise_broadcast_type;
    cfg.eltwise_broadcast_axis = FLAGS_eltwise_broadcast_axis;
    cfg.glu_type             = FLAGS_glu_type;
    cfg.split_axis           = FLAGS_split_axis;
    cfg.split_length         = FLAGS_split_length;
    cfg.gate_idx             = FLAGS_gate_idx;
    cfg.gather_axis          = FLAGS_gather_axis;
    cfg.batch_dim            = FLAGS_batch_dim;
    cfg.support_neg_ind      = FLAGS_support_neg_ind;
    cfg.head_cnt             = FLAGS_head_cnt;
    cfg.head_size            = FLAGS_head_size;
    cfg.rotary_ndims         = FLAGS_rotary_ndims;
    cfg.is_interleaved       = FLAGS_is_interleaved;
    cfg.is_chatglm           = FLAGS_is_chatglm;
    cfg.is_qwen              = FLAGS_is_qwen;
    cfg.input_trans0213      = FLAGS_input_trans0213;
    cfg.slice_start          = FLAGS_slice_start;
    cfg.slice_stop           = FLAGS_slice_stop;
    cfg.gather_rank          = FLAGS_gather_rank;
    cfg.offsets              = FLAGS_offsets;
    cfg.ss_begin             = FLAGS_ss_begin;
    cfg.ss_end               = FLAGS_ss_end;
    cfg.ss_strides           = FLAGS_ss_strides;
    cfg.begin_mask           = FLAGS_begin_mask;
    cfg.end_mask             = FLAGS_end_mask;
    cfg.shrink_axis_mask     = FLAGS_shrink_axis_mask;
    cfg.new_axis_mask        = FLAGS_new_axis_mask;
    cfg.concat_axis          = FLAGS_concat_axis;
    cfg.tile_repeats         = FLAGS_tile_repeats;
    cfg.across_spatial       = FLAGS_across_spatial;
    cfg.num_groups           = FLAGS_num_groups;
    cfg.levels               = FLAGS_levels;
    cfg.indices_rank         = FLAGS_indices_rank;
    cfg.resample_sizes       = FLAGS_resample_sizes;
    cfg.resample_mode        = FLAGS_resample_mode;
    cfg.permute_order        = FLAGS_permute_order;
    cfg.broadcast_axes       = FLAGS_broadcast_axes;
    cfg.broadcast_target     = FLAGS_broadcast_target;
    cfg.adaptive_pool_mode   = FLAGS_adaptive_pool_mode;
    cfg.adaptive_pool_out    = FLAGS_adaptive_pool_out;
    cfg.topk_mode            = FLAGS_topk_mode;
    cfg.top_k                = FLAGS_top_k;
    cfg.col2im_output_shape  = FLAGS_col2im_output_shape;
    cfg.col2im_kernel_shape  = FLAGS_col2im_kernel_shape;
    cfg.det_num_classes      = FLAGS_det_num_classes;
    cfg.det_keep_top_k       = FLAGS_det_keep_top_k;
    cfg.det_top_k            = FLAGS_det_top_k;
    cfg.det_nms_threshold    = static_cast<float>(FLAGS_det_nms_threshold);
    cfg.det_confidence_threshold = static_cast<float>(FLAGS_det_confidence_threshold);
    cfg.det_code_type        = FLAGS_det_code_type;
    cfg.det_share_location   = FLAGS_det_share_location;
    cfg.det_background_label_id = FLAGS_det_background_label_id;
    cfg.det_variance_encoded = FLAGS_det_variance_encoded;
    cfg.det_eta              = static_cast<float>(FLAGS_det_eta);
    cfg.det_prior_info_size  = FLAGS_det_prior_info_size;
    cfg.det_prior_coordinates_offset = FLAGS_det_prior_coordinates_offset;
    cfg.det_prior_is_normalized = FLAGS_det_prior_is_normalized;
    cfg.det_input_width      = FLAGS_det_input_width;
    cfg.det_input_height     = FLAGS_det_input_height;
    cfg.det_decrease_label_id = FLAGS_det_decrease_label_id;
    cfg.det_clip_before_nms  = FLAGS_det_clip_before_nms;
    cfg.det_clip_after_nms   = FLAGS_det_clip_after_nms;
    cfg.det_objectness_score = static_cast<float>(FLAGS_det_objectness_score);
    cfg.col2im_padding_begin = FLAGS_col2im_padding_begin;
    cfg.col2im_padding_end   = FLAGS_col2im_padding_end;
    cfg.scatter_mode         = FLAGS_scatter_mode;
    cfg.scatter_use_init_val = FLAGS_scatter_use_init_val;
    // New primitive-specific attributes
    cfg.cum_exclusive        = FLAGS_cum_exclusive;
    cfg.cum_reverse          = FLAGS_cum_reverse;
    cfg.block_size           = FLAGS_block_size;
    cfg.d2s_mode             = FLAGS_d2s_mode;
    cfg.grn_bias             = static_cast<float>(FLAGS_grn_bias);
    cfg.lrn_size             = FLAGS_lrn_size;
    cfg.lrn_k                = static_cast<float>(FLAGS_lrn_k);
    cfg.lrn_alpha            = static_cast<float>(FLAGS_lrn_alpha);
    cfg.lrn_beta             = static_cast<float>(FLAGS_lrn_beta);
    cfg.lrn_norm_region      = FLAGS_lrn_norm_region;
    cfg.roll_shift           = FLAGS_roll_shift;
    cfg.shuffle_group        = FLAGS_shuffle_group;
    cfg.border_mode          = FLAGS_border_mode;
    cfg.border_value         = static_cast<float>(FLAGS_border_value);
    cfg.reverse_mode         = FLAGS_reverse_mode;
    cfg.seq_axis             = FLAGS_seq_axis;
    cfg.one_hot_depth        = FLAGS_one_hot_depth;
    cfg.one_hot_on_value     = static_cast<float>(FLAGS_one_hot_on_value);
    cfg.one_hot_off_value    = static_cast<float>(FLAGS_one_hot_off_value);
    cfg.grid_mode            = FLAGS_grid_mode;
    cfg.grid_padding         = FLAGS_grid_padding;
    cfg.grid_align_corners   = FLAGS_grid_align_corners;
    cfg.roi_pooled_h         = FLAGS_roi_pooled_h;
    cfg.roi_pooled_w         = FLAGS_roi_pooled_w;
    cfg.roi_spatial_scale    = static_cast<float>(FLAGS_roi_spatial_scale);
    cfg.roi_sampling_ratio   = FLAGS_roi_sampling_ratio;
    cfg.eip_sizes            = FLAGS_eip_sizes;
    cfg.eip_strides          = FLAGS_eip_strides;
    cfg.eip_rates            = FLAGS_eip_rates;
    cfg.yolo_coords          = FLAGS_yolo_coords;
    cfg.yolo_classes         = FLAGS_yolo_classes;
    cfg.yolo_num             = FLAGS_yolo_num;
    cfg.yolo_do_softmax      = FLAGS_yolo_do_softmax;
    cfg.dft_inverse          = FLAGS_dft_inverse;
    cfg.range_start          = static_cast<float>(FLAGS_range_start);
    cfg.range_stop           = static_cast<float>(FLAGS_range_stop);
    cfg.range_step           = static_cast<float>(FLAGS_range_step);
    cfg.eye_diagonal         = FLAGS_eye_diagonal;
    cfg.nms_top_k            = FLAGS_nms_top_k;
    cfg.nms_iou_threshold    = static_cast<float>(FLAGS_nms_iou_threshold);
    cfg.nms_score_threshold  = static_cast<float>(FLAGS_nms_score_threshold);
    cfg.batch_file           = FLAGS_batch;
    cfg.parse_common();
    return cfg;
}

static bench_kernel::bench_stat report_unimplemented_result(const bench_kernel::bench_config& cfg,
                                                            const std::string& reason) {
    bench_kernel::bench_stat stat;
    stat.tests = 1;
    stat.unimplemented = 1;

    std::cout << "impl_info: unimplemented" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << cfg.test_index << ":" << bench_kernel::to_string(bench_kernel::test_status::unimplemented)
              << " (0.00 ms) " << reason
              << " __REPRO: " << cfg.repro_str() << std::endl;
    return stat;
}

static void list_devices() {
    try {
        cldnn::device_query query(
            cldnn::device_query::get_default_engine_type(),
            cldnn::device_query::get_default_runtime_type(),
            nullptr, nullptr, 0, -1, true);
        auto devices = query.get_available_devices();

        std::cout << "Available GPU devices:" << std::endl;
        for (const auto& dev : devices) {
            auto info = dev.second->get_info();
            std::cout << "  Device " << dev.first
                      << ": " << info.dev_name
                      << " (driver: " << info.driver_version << ")"
                      << std::endl;
        }
        if (devices.empty()) {
            std::cout << "  No GPU devices found." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error listing devices: " << e.what() << std::endl;
    }
}

static std::shared_ptr<cldnn::engine> create_engine(int device_id) {
    auto engine_type = cldnn::device_query::get_default_engine_type();
    auto runtime_type = cldnn::device_query::get_default_runtime_type();

    cldnn::device_query query(engine_type, runtime_type, nullptr, nullptr, 0, -1, true);
    auto devices = query.get_available_devices();

    if (devices.empty()) {
        throw std::runtime_error("No GPU devices found");
    }

    auto device_id_str = std::to_string(device_id);
    std::shared_ptr<cldnn::device> device;
    auto iter = devices.find(device_id_str);
    if (iter != devices.end()) {
        device = iter->second;
    } else {
        // Some runtimes expose non-numeric map keys; support numeric --device as ordinal index too.
        if (device_id >= 0 && static_cast<size_t>(device_id) < devices.size()) {
            auto it = devices.begin();
            std::advance(it, device_id);
            device = it->second;
        } else {
            device = devices.begin()->second;
        }
    }

    auto engine = cldnn::engine::create(engine_type, runtime_type, device);

#ifdef ENABLE_ONEDNN_FOR_GPU
    if (engine->get_device_info().supports_immad)
        engine->create_onednn_engine({});
#endif

    return engine;
}

// Process a batch file: each non-comment, non-empty line is either:
//   - A shapes string (e.g., "1x4096:4096x4096") - uses CLI flags for dt/impl/post-ops
//   - A full command (starts with "--") with all params inline:
//     e.g., "--conv --dt=f16:f16:f16 --shapes=1x3x224x224:32x3x3x3 --impl=onednn --attr-post-ops=relu"
static std::vector<std::string> read_batch_file(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open batch file: " + path);
    }
    std::string line;
    while (std::getline(file, line)) {
        // Trim
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        // Skip comments and empty
        if (line.empty() || line[0] == '#') continue;
        lines.push_back(line);
    }
    return lines;
}

// Parse a batch line with full command flags into a bench_config.
// Expected format: "--kernel_flag --dt=DT --shapes=SHAPES [--impl=IMPL] [--attr-post-ops=OPS] ..."
// Returns: {kernel_name, config} pair
static std::pair<std::string, bench_kernel::bench_config> parse_batch_cmd_line(const std::string& line) {
    bench_kernel::bench_config cfg;
    cfg.device               = FLAGS_device >= 0 ? FLAGS_device : 0;
    cfg.warmup_iters         = FLAGS_warmup;
    cfg.perf_iters           = FLAGS_iters;
    cfg.max_ms_per_prb       = FLAGS_max_ms;
    cfg.verbose              = FLAGS_verbose;
    cfg.mode_str             = FLAGS_mode;

    std::string kernel_name;

    // Map short kernel names to full names
    static const std::map<std::string, std::string> SHORT_TO_FULL = {
        {"fc", "fully_connected"}, {"gemm", "gemm"}, {"conv", "convolution"},
        {"sdpa", "scaled_dot_product_attention"}, {"eltwise", "eltwise"},
        {"softmax", "softmax"}, {"reduce", "reduce"}, {"activation", "activation"},
        {"pooling", "pooling"}, {"mvn", "mvn"}, {"reorder", "reorder"},
        {"permute", "permute"}, {"concatenation", "concatenation"},
        {"deconv", "deconvolution"}, {"resample", "resample"}, {"rms", "rms"},
        {"swiglu", "swiglu"}, {"dynamic_quantize", "dynamic_quantize"},
        {"swiglu_standalone", "swiglu"},
        {"gather", "gather"}, {"crop", "crop"}, {"rope", "rope"},
        {"strided_slice", "strided_slice"}, {"broadcast", "broadcast"},
        {"select", "select"}, {"scatter_update", "scatter_update"},
        {"tile", "tile"}, {"normalize", "normalize"},
        {"gather_elements", "gather_elements"}, {"scatter_nd_update", "scatter_nd_update"},
        {"scatter_elements_update", "scatter_elements_update"},
        {"group_normalization", "group_normalization"}, {"quantize", "quantize"},
        {"adaptive_pooling", "adaptive_pooling"}, {"arg_max_min", "arg_max_min"},
        {"col2im", "col2im"}, {"detection_output", "detection_output"},
        {"cum_sum", "cum_sum"}, {"depth_to_space", "depth_to_space"},
        {"space_to_depth", "space_to_depth"}, {"grn", "grn"}, {"lrn", "lrn"},
        {"reorg_yolo", "reorg_yolo"}, {"roll", "roll"},
        {"shuffle_channels", "shuffle_channels"}, {"non_zero", "non_zero"},
        {"bucketize", "bucketize"}, {"gather_nd", "gather_nd"},
        {"gather_tree", "gather_tree"}, {"one_hot", "one_hot"},
        {"reverse", "reverse"}, {"reverse_sequence", "reverse_sequence"},
        {"search_sorted", "search_sorted"}, {"slice", "slice"},
        {"border", "border"}, {"batch_to_space", "batch_to_space"},
        {"space_to_batch", "space_to_batch"}, {"grid_sample", "grid_sample"},
        {"roi_align", "roi_align"}, {"roi_pooling", "roi_pooling"},
        {"extract_image_patches", "extract_image_patches"},
        {"convert_color", "convert_color"}, {"region_yolo", "region_yolo"},
        {"dft", "dft"}, {"range", "range"}, {"random_uniform", "random_uniform"},
        {"eye", "eye"}, {"embedding_bag", "embedding_bag"},
        {"fake_convert", "fake_convert"}, {"ctc_greedy_decoder", "ctc_greedy_decoder"},
        {"non_max_suppression", "non_max_suppression"},
        {"prior_box", "prior_box"}, {"proposal", "proposal"},
        {"ctc_loss", "ctc_loss"}, {"stft", "STFT"}, {"istft", "ISTFT"},
        {"lstm_cell", "lstm_cell"}, {"matrix_nms", "matrix_nms"},
        {"multiclass_nms", "multiclass_nms"}, {"multinomial", "multinomial"},
        {"segment_max", "segment_max"},
        {"sparse_fill_empty_rows", "sparse_fill_empty_rows"},
        {"unique_count", "unique_count"}, {"unique_gather", "unique_gather"},
        {"generate_proposals", "generate_proposals"},
        {"exp_detectron_detection_output", "experimental_detectron_detection_output"},
        {"experimental_detectron_detection_output", "experimental_detectron_detection_output"},
        {"exp_detectron_generate_proposals", "experimental_detectron_generate_proposals_single_image"},
        {"experimental_detectron_generate_proposals_single_image", "experimental_detectron_generate_proposals_single_image"},
        {"exp_detectron_prior_grid_gen", "experimental_detectron_prior_grid_generator"},
        {"experimental_detectron_prior_grid_generator", "experimental_detectron_prior_grid_generator"},
        {"exp_detectron_roi_feature", "experimental_detectron_roi_feature_extractor"},
        {"experimental_detectron_roi_feature_extractor", "experimental_detectron_roi_feature_extractor"},
        {"exp_detectron_topk_rois", "experimental_detectron_topk_rois"},
        {"experimental_detectron_topk_rois", "experimental_detectron_topk_rois"},
    };

    // Helper to parse --key=value tokens
    auto try_parse = [](const std::string& token, const std::string& prefix, std::string& out) -> bool {
        if (token.find(prefix) == 0) { out = token.substr(prefix.size()); return true; }
        return false;
    };
    auto try_parse_int = [](const std::string& token, const std::string& prefix, int& out) -> bool {
        if (token.find(prefix) == 0) { out = std::stoi(token.substr(prefix.size())); return true; }
        return false;
    };
    auto try_parse_float = [](const std::string& token, const std::string& prefix, float& out) -> bool {
        if (token.find(prefix) == 0) { out = std::stof(token.substr(prefix.size())); return true; }
        return false;
    };

    // Tokenize the line
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        if (token.find("--dt=") == 0) {
            cfg.dt_str = token.substr(5);
        } else if (token.find("--shapes=") == 0) {
            cfg.shapes_str = token.substr(9);
        } else if (token.find("--impl=") == 0) {
            cfg.impl_str = token.substr(7);
        } else if (token.find("--force_impl=") == 0) {
            cfg.force_impl_str = token.substr(13);
        } else if (try_parse(token, "--in_layouts=", cfg.in_layouts_str)) {
        } else if (try_parse(token, "--out_layouts=", cfg.out_layouts_str)) {
        } else if (token.find("--attr-post-ops=") == 0) {
            cfg.attr_post_ops_str = token.substr(16);
        } else if (token.find("--attr-scales=") == 0) {
            cfg.attr_scales_str = token.substr(14);
        } else if (token.find("--attr-zero-points=") == 0) {
            cfg.attr_zero_points_str = token.substr(19);
        } else if (token == "--truncate") {
            cfg.truncate = true;
        } else if (token.find("--mode=") == 0) {
            cfg.mode_str = token.substr(7);
        // Primitive-specific attribute flags
        } else if (try_parse_int(token, "--transpose_a=", cfg.transpose_a)) {
        } else if (try_parse_int(token, "--transpose_b=", cfg.transpose_b)) {
        } else if (try_parse(token, "--gemm_order0=", cfg.gemm_order0)) {
        } else if (try_parse(token, "--gemm_order1=", cfg.gemm_order1)) {
        } else if (try_parse(token, "--gemm_order_out=", cfg.gemm_order_out)) {
        } else if (try_parse_int(token, "--is_causal=", cfg.is_causal)) {
        } else if (try_parse(token, "--order_q=", cfg.order_q)) {
        } else if (try_parse(token, "--order_k=", cfg.order_k)) {
        } else if (try_parse(token, "--order_v=", cfg.order_v)) {
        } else if (try_parse(token, "--order_out=", cfg.order_out)) {
        } else if (try_parse(token, "--scale_val=", cfg.scale_val)) {
        } else if (try_parse_int(token, "--compressed=", cfg.compressed)) {
        } else if (try_parse_int(token, "--dynamic_quantized=", cfg.dynamic_quantized)) {
        } else if (try_parse_int(token, "--dynamic_quantized_zp=", cfg.dynamic_quantized_zp)) {
        } else if (try_parse_int(token, "--dynamic_quantized_precomputed_reduction=", cfg.dynamic_quantized_precomputed_reduction)) {
        } else if (try_parse_int(token, "--fc_input_size=", cfg.fc_input_size)) {
        } else if (try_parse_int(token, "--fc_weights_rank=", cfg.fc_weights_rank)) {
        } else if (try_parse_int(token, "--groups=", cfg.groups)) {
        } else if (try_parse(token, "--strides=", cfg.strides)) {
        } else if (try_parse(token, "--dilations=", cfg.dilations)) {
        } else if (try_parse(token, "--padding_begin=", cfg.padding_begin)) {
        } else if (try_parse(token, "--padding_end=", cfg.padding_end)) {
        } else if (try_parse_int(token, "--grouped_weights_shape=", cfg.grouped_weights_shape)) {
        } else if (try_parse_int(token, "--pool_mode=", cfg.pool_mode)) {
        } else if (try_parse(token, "--kernel=", cfg.pool_kernel)) {
        } else if (try_parse(token, "--pool_strides=", cfg.pool_strides)) {
        } else if (try_parse(token, "--pads_begin=", cfg.pads_begin)) {
        } else if (try_parse(token, "--pads_end=", cfg.pads_end)) {
        } else if (try_parse_int(token, "--rounding_type=", cfg.rounding_type)) {
        } else if (try_parse_int(token, "--reduce_mode=", cfg.reduce_mode)) {
        } else if (try_parse_int(token, "--keep_dims=", cfg.keep_dims)) {
        } else if (try_parse(token, "--reduce_axes=", cfg.reduce_axes)) {
        } else if (try_parse_int(token, "--axis=", cfg.axis)) {
        } else if (try_parse_int(token, "--normalize_variance=", cfg.normalize_variance)) {
        } else if (try_parse(token, "--epsilon=", cfg.epsilon)) {
        } else if (try_parse_int(token, "--eps_inside_sqrt=", cfg.eps_inside_sqrt)) {
        } else if (try_parse(token, "--mvn_reduction_axes=", cfg.mvn_reduction_axes)) {
        } else if (try_parse_int(token, "--eltwise_mode=", cfg.eltwise_mode)) {
        } else if (try_parse_int(token, "--pythondiv=", cfg.pythondiv)) {
        } else if (try_parse(token, "--eltwise_coefficients=", cfg.eltwise_coefficients)) {
        } else if (try_parse(token, "--eltwise_stride=", cfg.eltwise_stride)) {
        } else if (try_parse(token, "--eltwise_broadcast_type=", cfg.eltwise_broadcast_type)) {
        } else if (try_parse_int(token, "--eltwise_broadcast_axis=", cfg.eltwise_broadcast_axis)) {
        } else if (try_parse_int(token, "--glu_type=", cfg.glu_type)) {
        } else if (try_parse_int(token, "--split_axis=", cfg.split_axis)) {
        } else if (try_parse_int(token, "--split_length=", cfg.split_length)) {
        } else if (try_parse_int(token, "--gate_idx=", cfg.gate_idx)) {
        } else if (try_parse_int(token, "--gather_axis=", cfg.gather_axis)) {
        } else if (try_parse_int(token, "--batch_dim=", cfg.batch_dim)) {
        } else if (try_parse_int(token, "--support_neg_ind=", cfg.support_neg_ind)) {
        } else if (try_parse_int(token, "--head_cnt=", cfg.head_cnt)) {
        } else if (try_parse_int(token, "--head_size=", cfg.head_size)) {
        } else if (try_parse_int(token, "--rotary_ndims=", cfg.rotary_ndims)) {
        } else if (try_parse_int(token, "--is_interleaved=", cfg.is_interleaved)) {
        } else if (try_parse_int(token, "--is_chatglm=", cfg.is_chatglm)) {
        } else if (try_parse_int(token, "--is_qwen=", cfg.is_qwen)) {
        } else if (try_parse_int(token, "--input_trans0213=", cfg.input_trans0213)) {
        } else if (try_parse_int(token, "--slice_start=", cfg.slice_start)) {
        } else if (try_parse_int(token, "--slice_stop=", cfg.slice_stop)) {
        } else if (try_parse_int(token, "--gather_rank=", cfg.gather_rank)) {
        } else if (try_parse(token, "--offsets=", cfg.offsets)) {
        } else if (try_parse(token, "--ss_begin=", cfg.ss_begin)) {
        } else if (try_parse(token, "--ss_end=", cfg.ss_end)) {
        } else if (try_parse(token, "--ss_strides=", cfg.ss_strides)) {
        } else if (try_parse(token, "--begin_mask=", cfg.begin_mask)) {
        } else if (try_parse(token, "--end_mask=", cfg.end_mask)) {
        } else if (try_parse(token, "--shrink_axis_mask=", cfg.shrink_axis_mask)) {
        } else if (try_parse(token, "--new_axis_mask=", cfg.new_axis_mask)) {
        } else if (try_parse_int(token, "--concat_axis=", cfg.concat_axis)) {
        } else if (try_parse(token, "--tile_repeats=", cfg.tile_repeats)) {
        } else if (try_parse_int(token, "--across_spatial=", cfg.across_spatial)) {
        } else if (try_parse_int(token, "--num_groups=", cfg.num_groups)) {
        } else if (try_parse_int(token, "--levels=", cfg.levels)) {
        } else if (try_parse_int(token, "--indices_rank=", cfg.indices_rank)) {
        } else if (try_parse(token, "--resample_sizes=", cfg.resample_sizes)) {
        } else if (try_parse_int(token, "--resample_mode=", cfg.resample_mode)) {
        } else if (try_parse(token, "--permute_order=", cfg.permute_order)) {
        } else if (try_parse(token, "--broadcast_axes=", cfg.broadcast_axes)) {
        } else if (try_parse(token, "--broadcast_target=", cfg.broadcast_target)) {
        } else if (try_parse_int(token, "--adaptive_pool_mode=", cfg.adaptive_pool_mode)) {
        } else if (try_parse(token, "--adaptive_pool_out=", cfg.adaptive_pool_out)) {
        } else if (try_parse_int(token, "--topk_mode=", cfg.topk_mode)) {
        } else if (try_parse_int(token, "--top_k=", cfg.top_k)) {
        } else if (try_parse(token, "--col2im_output_shape=", cfg.col2im_output_shape)) {
        } else if (try_parse(token, "--col2im_kernel_shape=", cfg.col2im_kernel_shape)) {
        } else if (try_parse_int(token, "--det_num_classes=", cfg.det_num_classes)) {
        } else if (try_parse_int(token, "--det_keep_top_k=", cfg.det_keep_top_k)) {
        } else if (try_parse_int(token, "--det_top_k=", cfg.det_top_k)) {
        } else if (try_parse_float(token, "--det_nms_threshold=", cfg.det_nms_threshold)) {
        } else if (try_parse_float(token, "--det_confidence_threshold=", cfg.det_confidence_threshold)) {
        } else if (try_parse_int(token, "--det_code_type=", cfg.det_code_type)) {
        } else if (try_parse_int(token, "--det_share_location=", cfg.det_share_location)) {
        } else if (try_parse_int(token, "--det_background_label_id=", cfg.det_background_label_id)) {
        } else if (try_parse_int(token, "--det_variance_encoded=", cfg.det_variance_encoded)) {
        } else if (try_parse_float(token, "--det_eta=", cfg.det_eta)) {
        } else if (try_parse_int(token, "--det_prior_info_size=", cfg.det_prior_info_size)) {
        } else if (try_parse_int(token, "--det_prior_coordinates_offset=", cfg.det_prior_coordinates_offset)) {
        } else if (try_parse_int(token, "--det_prior_is_normalized=", cfg.det_prior_is_normalized)) {
        } else if (try_parse_int(token, "--det_input_width=", cfg.det_input_width)) {
        } else if (try_parse_int(token, "--det_input_height=", cfg.det_input_height)) {
        } else if (try_parse_int(token, "--det_decrease_label_id=", cfg.det_decrease_label_id)) {
        } else if (try_parse_int(token, "--det_clip_before_nms=", cfg.det_clip_before_nms)) {
        } else if (try_parse_int(token, "--det_clip_after_nms=", cfg.det_clip_after_nms)) {
        } else if (try_parse_float(token, "--det_objectness_score=", cfg.det_objectness_score)) {
        } else if (try_parse(token, "--col2im_padding_begin=", cfg.col2im_padding_begin)) {
        } else if (try_parse(token, "--col2im_padding_end=", cfg.col2im_padding_end)) {
        } else if (try_parse_int(token, "--scatter_mode=", cfg.scatter_mode)) {
        } else if (try_parse_int(token, "--scatter_use_init_val=", cfg.scatter_use_init_val)) {
        // New primitive-specific attribute flags
        } else if (try_parse_int(token, "--cum_exclusive=", cfg.cum_exclusive)) {
        } else if (try_parse_int(token, "--cum_reverse=", cfg.cum_reverse)) {
        } else if (try_parse_int(token, "--block_size=", cfg.block_size)) {
        } else if (try_parse_int(token, "--d2s_mode=", cfg.d2s_mode)) {
        } else if (try_parse_float(token, "--grn_bias=", cfg.grn_bias)) {
        } else if (try_parse_int(token, "--lrn_size=", cfg.lrn_size)) {
        } else if (try_parse_float(token, "--lrn_k=", cfg.lrn_k)) {
        } else if (try_parse_float(token, "--lrn_alpha=", cfg.lrn_alpha)) {
        } else if (try_parse_float(token, "--lrn_beta=", cfg.lrn_beta)) {
        } else if (try_parse_int(token, "--lrn_norm_region=", cfg.lrn_norm_region)) {
        } else if (try_parse(token, "--roll_shift=", cfg.roll_shift)) {
        } else if (try_parse_int(token, "--shuffle_group=", cfg.shuffle_group)) {
        } else if (try_parse_int(token, "--border_mode=", cfg.border_mode)) {
        } else if (try_parse_float(token, "--border_value=", cfg.border_value)) {
        } else if (try_parse_int(token, "--reverse_mode=", cfg.reverse_mode)) {
        } else if (try_parse_int(token, "--seq_axis=", cfg.seq_axis)) {
        } else if (try_parse_int(token, "--one_hot_depth=", cfg.one_hot_depth)) {
        } else if (try_parse_float(token, "--one_hot_on_value=", cfg.one_hot_on_value)) {
        } else if (try_parse_float(token, "--one_hot_off_value=", cfg.one_hot_off_value)) {
        } else if (try_parse_int(token, "--grid_mode=", cfg.grid_mode)) {
        } else if (try_parse_int(token, "--grid_padding=", cfg.grid_padding)) {
        } else if (try_parse_int(token, "--grid_align_corners=", cfg.grid_align_corners)) {
        } else if (try_parse_int(token, "--roi_pooled_h=", cfg.roi_pooled_h)) {
        } else if (try_parse_int(token, "--roi_pooled_w=", cfg.roi_pooled_w)) {
        } else if (try_parse_float(token, "--roi_spatial_scale=", cfg.roi_spatial_scale)) {
        } else if (try_parse_int(token, "--roi_sampling_ratio=", cfg.roi_sampling_ratio)) {
        } else if (try_parse(token, "--eip_sizes=", cfg.eip_sizes)) {
        } else if (try_parse(token, "--eip_strides=", cfg.eip_strides)) {
        } else if (try_parse(token, "--eip_rates=", cfg.eip_rates)) {
        } else if (try_parse_int(token, "--yolo_coords=", cfg.yolo_coords)) {
        } else if (try_parse_int(token, "--yolo_classes=", cfg.yolo_classes)) {
        } else if (try_parse_int(token, "--yolo_num=", cfg.yolo_num)) {
        } else if (try_parse_int(token, "--yolo_do_softmax=", cfg.yolo_do_softmax)) {
        } else if (try_parse_int(token, "--dft_inverse=", cfg.dft_inverse)) {
        } else if (try_parse_float(token, "--range_start=", cfg.range_start)) {
        } else if (try_parse_float(token, "--range_stop=", cfg.range_stop)) {
        } else if (try_parse_float(token, "--range_step=", cfg.range_step)) {
        } else if (try_parse_int(token, "--eye_diagonal=", cfg.eye_diagonal)) {
        } else if (try_parse_int(token, "--nms_top_k=", cfg.nms_top_k)) {
        } else if (try_parse_float(token, "--nms_iou_threshold=", cfg.nms_iou_threshold)) {
        } else if (try_parse_float(token, "--nms_score_threshold=", cfg.nms_score_threshold)) {
        } else if (token.find("--device=") == 0) {
            cfg.device = std::stoi(token.substr(9));
        } else if (token.find("--") == 0) {
            // Kernel type flag
            std::string kname = token.substr(2);
            auto it = SHORT_TO_FULL.find(kname);
            if (it != SHORT_TO_FULL.end()) {
                kernel_name = it->second;
            } else {
                kernel_name = kname;
            }
        }
    }

    cfg.kernel_name = kernel_name;
    cfg.parse_common();
    return {kernel_name, cfg};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    gflags::SetUsageMessage(
        "bench_kernel - OpenVINO GPU kernel-level benchmark tool\n"
        "\n"
        "Usage:\n"
        "  ./ov_gpu_bench_kernel --fc --dt=f16 1x4096:4096x4096\n"
        "  ./ov_gpu_bench_kernel --gemm --dt=f16 --attr-post-ops=relu 64x128:128x256\n"
        "  ./ov_gpu_bench_kernel --fc --dt=f16:i4:f16 --attr-scales=wei:per_ocic:f16:1x128 1x4096:4096x4096\n"
        "  ./ov_gpu_bench_kernel --list-devices\n"
        "  ./ov_gpu_bench_kernel --list-kernels\n"
    );
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // --list-devices
    if (FLAGS_list_devices) {
        list_devices();
        return 0;
    }

    // --list-kernels
    if (FLAGS_list_kernels) {
        bench_kernel::kernel_registry::instance().list_all();
        return 0;
    }

    // Detect kernel type (may be empty when batch file has full-command lines)
    std::string kernel_name = detect_kernel_name();
    if (kernel_name.empty() && FLAGS_batch.empty()) {
        std::cerr << "Error: No kernel type specified. Use --fc, --gemm, --conv, etc." << std::endl;
        std::cerr << "Or use --batch=file.txt with full-command lines." << std::endl;
        std::cerr << "Run with --list-kernels to see available kernels." << std::endl;
        return 1;
    }

    // Collect shapes: --shapes= flag takes priority, then positional arg
    std::string shapes_str = FLAGS_shapes;
    if (shapes_str.empty()) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            // Skip gflags-style args
            if (arg.find("--") == 0 || arg.find("-") == 0) continue;
            shapes_str = arg;
            break;
        }
    }

    // Check kernel is registered (skip check if batch file will provide kernel names)
    auto& registry = bench_kernel::kernel_registry::instance();
    if (!kernel_name.empty() && !registry.has(kernel_name)) {
        auto cfg = build_config(kernel_name, shapes_str);
        cfg.test_index = 0;
        auto total_stat = report_unimplemented_result(cfg,
                            "Kernel '" + kernel_name + "' is not implemented in bench_kernel");
        total_stat.print();
        return 0;
    }

    // Create engine (treat -1 as 0)
    int device_idx = FLAGS_device >= 0 ? FLAGS_device : 0;
    std::shared_ptr<cldnn::engine> engine;
    try {
        engine = create_engine(device_idx);
    } catch (const std::exception& e) {
        std::cerr << "Error creating GPU engine: " << e.what() << std::endl;
        return 1;
    }

    auto dev_info = engine->get_device_info();
    if (FLAGS_verbose >= 2) {
        std::cout << "bench_kernel: device=" << dev_info.dev_name
                  << " (driver: " << dev_info.driver_version << ")" << std::endl;
    }

    bench_kernel::bench_stat total_stat;
    int test_counter = 0;

    // If batch file is specified
    if (!FLAGS_batch.empty()) {
        auto lines = read_batch_file(FLAGS_batch);
        std::cout << "Processing batch file: " << FLAGS_batch
                  << " (" << lines.size() << " problems)" << std::endl;

        for (const auto& line : lines) {
            // Check if line contains full command flags (starts with --)
            if (line.find("--") == 0) {
                // Full command line with kernel type, dt, shapes, impl, etc.
                auto [cmd_kernel, cmd_cfg] = parse_batch_cmd_line(line);
                if (cmd_kernel.empty()) {
                    std::cerr << "Warning: Could not parse batch line: " << line << std::endl;
                    continue;
                }
                if (!registry.has(cmd_kernel)) {
                    cmd_cfg.test_index = test_counter++;
                    auto stat = report_unimplemented_result(
                        cmd_cfg,
                        "Kernel '" + cmd_kernel + "' is not implemented in bench_kernel");
                    total_stat.merge(stat);
                    continue;
                }
                cmd_cfg.test_index = test_counter++;
                if (cmd_cfg.verbose >= 2) cmd_cfg.print();

                auto k = registry.create(cmd_kernel);
                auto stat = k->run(*engine, cmd_cfg);
                total_stat.merge(stat);
            } else {
                // Simple shapes string — use CLI kernel_name and flags
                auto cfg = build_config(kernel_name, line);
                cfg.test_index = test_counter++;
                if (cfg.verbose >= 2) cfg.print();

                auto k = registry.create(kernel_name);
                auto stat = k->run(*engine, cfg);
                total_stat.merge(stat);
            }
        }
    } else {
        // Single problem
        if (shapes_str.empty()) {
            std::cerr << "Error: No shapes specified. Provide shapes as positional argument." << std::endl;
            std::cerr << "Example: ./ov_gpu_bench_kernel --fc --dt=f16 1x4096:4096x4096" << std::endl;
            return 1;
        }

        auto cfg = build_config(kernel_name, shapes_str);
        cfg.test_index = test_counter++;
        if (cfg.verbose >= 2) cfg.print();

        auto kernel = registry.create(kernel_name);
        total_stat = kernel->run(*engine, cfg);
    }

    // Print summary
    auto summary_mode = bench_kernel::str2mode(FLAGS_mode);
    int print_mode = 0;  // perf
    if (summary_mode == bench_kernel::bench_mode::acc) print_mode = 1;
    else if (summary_mode == bench_kernel::bench_mode::corr_perf) print_mode = 2;
    total_stat.print(print_mode);

    return total_stat.failed > 0 ? 1 : 0;
}
