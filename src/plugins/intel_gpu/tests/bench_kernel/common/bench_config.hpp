// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdlib>

#include "bench_types.hpp"

namespace bench_kernel {

// Helper: parse colon-separated int vector (e.g. "0:2:1:3" → {0,2,1,3})
inline std::vector<int64_t> parse_colon_vec(const std::string& s) {
    std::vector<int64_t> v;
    if (s.empty()) return v;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, ':')) {
        v.push_back(std::stoll(tok));
    }
    return v;
}

// Helper: parse 'x'-separated size vector (e.g. "2x2" → {2,2})
inline std::vector<size_t> parse_x_vec(const std::string& s) {
    std::vector<size_t> v;
    if (s.empty()) return v;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, 'x')) {
        v.push_back(std::stoull(tok));
    }
    return v;
}

// Helper: parse 'x'-separated signed vector (e.g. "1x1" → {1,1})
inline std::vector<ptrdiff_t> parse_x_svec(const std::string& s) {
    std::vector<ptrdiff_t> v;
    if (s.empty()) return v;
    std::istringstream iss(s);
    std::string tok;
    while (std::getline(iss, tok, 'x')) {
        v.push_back(std::stoll(tok));
    }
    return v;
}

// ============================================================================
// Benchmark execution mode
// ============================================================================

enum class bench_mode {
    perf,       // p : Performance measurement (warmup=5, iters=20)
    acc,        // c : Correctness / accuracy check against CPU reference
    fast_perf,  // f : Fast performance (warmup=1, iters=3)
    corr_perf,  // cp: Correctness + performance
    run,        // r : Run-only (warmup=0, iters=1, just execute once)
};

inline bench_mode str2mode(const std::string& s) {
    if (s == "c" || s == "acc" || s == "accuracy")  return bench_mode::acc;
    if (s == "f" || s == "fast")                     return bench_mode::fast_perf;
    if (s == "cp" || s == "both" || s == "all")      return bench_mode::corr_perf;
    if (s == "r" || s == "run")                      return bench_mode::run;
    return bench_mode::perf;  // default: "p"
}

inline std::string mode2str(bench_mode m) {
    switch (m) {
        case bench_mode::acc:       return "c";
        case bench_mode::fast_perf: return "f";
        case bench_mode::corr_perf: return "cp";
        case bench_mode::run:       return "r";
        default:                    return "p";
    }
}

// ============================================================================
// Global bench_kernel configuration (parsed from CLI)
// ============================================================================

struct bench_config {
    // Execution control
    int    device           = 0;         // GPU device index (--device=N)
    int    warmup_iters     = 5;         // warmup iterations
    int    perf_iters       = 20;        // measurement iterations
    double max_ms_per_prb   = 3000.0;    // max time per problem (ms)
    int    verbose          = 0;         // verbosity level (0=silent, 1=summary, 2=detail)
    int    test_index       = 0;         // test counter for benchdnn-style output

    // What to do
    bool   list_devices     = false;     // --list-devices
    bool   perf_mode        = true;      // measure performance

    // Benchmark mode
    bench_mode mode         = bench_mode::perf;
    std::string mode_str;               // --mode=c|p|f|cp|r

    // Common kernel parameters
    std::string  kernel_name;            // --fc, --gemm, --conv, ...
    std::string  dt_str;                 // --dt=f16:i4:f16
    std::string  impl_str;              // --impl=ocl
    std::string  force_impl_str;         // --force_impl=kernel_name (requires --impl=ocl)
    std::string  shapes_str;            // positional: 1x4096:4096x4096

    // Layout format strings (from BenchVerbose log)
    std::string  in_layouts_str;          // --in_layouts=bfyx,b_fs_yx_fsv16  (comma-separated per input)
    std::string  out_layouts_str;         // --out_layouts=b_fs_yx_fsv16  (comma-separated per output)

    // Attribute strings (parsed later by individual kernels)
    std::string  attr_scales_str;       // --attr-scales=...
    std::string  attr_zero_points_str;  // --attr-zero-points=...
    std::string  attr_post_ops_str;     // --attr-post-ops=...

    // Reorder-specific
    bool         truncate = false;      // --truncate (Convert op truncation mode)

    // Gemm-specific
    int          transpose_a = 0;       // --transpose_a=0|1
    int          transpose_b = 0;       // --transpose_b=0|1
    std::string  gemm_order0;            // --gemm_order0=0:1:2:3 (input0 transpose order)
    std::string  gemm_order1;            // --gemm_order1=0:1:2:3 (input1 transpose order)
    std::string  gemm_order_out;         // --gemm_order_out=0:1:2:3 (output transpose order)

    // SDPA-specific
    int          is_causal = -1;        // --is_causal=0|1 (-1=auto from mask)
    std::string  order_q;               // --order_q=0:2:1:3
    std::string  order_k;               // --order_k=0:2:1:3
    std::string  order_v;               // --order_v=0:2:1:3
    std::string  order_out;             // --order_out=0:2:1:3
    std::string  scale_val;             // --scale_val=0.088388

    // Convolution-specific
    int          groups = 1;            // --groups=N
    std::string  strides;               // --strides=2x2
    std::string  dilations;             // --dilations=1x1
    std::string  padding_begin;         // --padding_begin=1x1
    std::string  padding_end;           // --padding_end=1x1
    int          grouped_weights_shape = 0;  // --grouped_weights_shape=0|1

    // Pooling-specific
    int          pool_mode = 0;         // --pool_mode=N (0=max, 1=avg, 2=avg_no_pad)
    std::string  pool_kernel;           // --kernel=2x2
    std::string  pool_strides;          // --pool_strides=2x2
    std::string  pads_begin;            // --pads_begin=0x0
    std::string  pads_end;              // --pads_end=0x0
    int          rounding_type = 0;     // --rounding_type=0|1 (0=floor, 1=ceil)

    // Reduce-specific
    int          reduce_mode = 1;       // --reduce_mode=N (1=mean)
    int          keep_dims = 1;         // --keep_dims=0|1
    std::string  reduce_axes;           // --reduce_axes=1:2:3

    // Softmax-specific
    int          axis = -1;             // --axis=N (-1=last dim)

    // MVN-specific
    int          normalize_variance = 1;  // --normalize_variance=0|1
    std::string  epsilon;               // --epsilon=1e-6
    int          eps_inside_sqrt = 0;   // --eps_inside_sqrt=0|1

    // Eltwise-specific
    int          eltwise_mode = -1;     // --eltwise_mode=N (-1=from post-ops)
    int          pythondiv = 0;         // --pythondiv=0|1

    // SwiGLU-specific
    int          glu_type = 0;          // --glu_type=N (0=Swish)
    int          split_axis = -1;       // --split_axis=N (-1=last)
    int          split_length = -1;     // --split_length=N (-1=auto)
    int          gate_idx = 0;          // --gate_idx=0|1

    // Gather-specific
    int          gather_axis = 0;       // --gather_axis=N
    int          batch_dim = 0;         // --batch_dim=N
    int          support_neg_ind = 0;   // --support_neg_ind=0|1

    // RoPE-specific
    int          head_cnt = 0;          // --head_cnt=N
    int          head_size = 0;         // --head_size=N
    int          rotary_ndims = 0;      // --rotary_ndims=N
    int          is_interleaved = 0;    // --is_interleaved=0|1
    int          is_chatglm = 0;        // --is_chatglm=0|1
    int          is_qwen = 0;           // --is_qwen=0|1
    int          input_trans0213 = 1;   // --input_trans0213=0|1
    int          slice_start = 0;       // --slice_start=N
    int          slice_stop = 0;        // --slice_stop=N
    int          gather_rank = 0;       // --gather_rank=N

    // Crop-specific
    std::string  offsets;               // --offsets=0:0:0:0

    // StridedSlice-specific
    std::string  ss_begin;              // --ss_begin=0:0:0
    std::string  ss_end;                // --ss_end=1:2:3
    std::string  ss_strides;            // --ss_strides=1:1:1
    std::string  begin_mask;            // --begin_mask=0:0:0
    std::string  end_mask;              // --end_mask=0:0:0
    std::string  shrink_axis_mask;      // --shrink_axis_mask=0:0:0
    std::string  new_axis_mask;         // --new_axis_mask=0:0:0

    // Concatenation-specific
    int          concat_axis = 1;       // --concat_axis=N

    // Tile-specific
    std::string  tile_repeats;          // --tile_repeats=2:2:1:1

    // Normalize-specific
    int          across_spatial = 0;    // --across_spatial=0|1

    // GroupNormalization-specific
    int          num_groups = 1;        // --num_groups=N

    // Quantize-specific
    int          levels = 0;            // --levels=N (256=FakeQuantize)

    // ScatterNDUpdate-specific
    int          indices_rank = 0;      // --indices_rank=N

    // Resample-specific
    std::string  resample_sizes;        // --resample_sizes=1:64:64:64
    int          resample_mode = 0;     // --resample_mode=0|1|2 (nearest/linear/cubic)

    // Permute-specific
    std::string  permute_order;         // --permute_order=0:2:1:3

    // Broadcast-specific
    std::string  broadcast_axes;        // --broadcast_axes=0:1
    std::string  broadcast_target;      // --broadcast_target=1:32:128:128

    // AdaptivePooling-specific
    int          adaptive_pool_mode = 0; // --adaptive_pool_mode=0|1 (0=avg, 1=max)
    std::string  adaptive_pool_out;      // --adaptive_pool_out=1:64:7:7

    // ArgMaxMin (TopK)-specific
    int          topk_mode = 0;          // --topk_mode=0|1 (0=max, 1=min)
    int          top_k = 1;             // --top_k=N

    // Col2Im-specific
    std::string  col2im_output_shape;    // --col2im_output_shape=4x4
    std::string  col2im_kernel_shape;    // --col2im_kernel_shape=3x3

    // DetectionOutput-specific
    int          det_num_classes = 21;   // --det_num_classes=N
    int          det_keep_top_k = 200;   // --det_keep_top_k=N
    int          det_top_k = -1;        // --det_top_k=N
    float        det_nms_threshold = 0.45f;  // --det_nms_threshold=0.45
    float        det_confidence_threshold = 0.01f;  // --det_confidence_threshold=0.01
    int          det_code_type = 1;      // --det_code_type=0|1|2 (corner/center_size/corner_size)
    int          det_share_location = 1; // --det_share_location=0|1
    int          det_background_label_id = 0;  // --det_background_label_id=N
    int          det_variance_encoded = 0;  // --det_variance_encoded=0|1

    // CumSum-specific
    int          cum_exclusive = 0;      // --cum_exclusive=0|1
    int          cum_reverse = 0;        // --cum_reverse=0|1

    // DepthToSpace / SpaceToDepth / ReorgYolo -specific
    int          block_size = 0;         // --block_size=N
    int          d2s_mode = 0;           // --d2s_mode=0|1 (0=blocks_first, 1=depth_first)

    // GRN-specific
    float        grn_bias = 1e-6f;       // --grn_bias=1e-6

    // LRN-specific
    int          lrn_size = 5;           // --lrn_size=N
    float        lrn_k = 1.0f;          // --lrn_k=1.0
    float        lrn_alpha = 0.0001f;   // --lrn_alpha=0.0001
    float        lrn_beta = 0.75f;      // --lrn_beta=0.75
    int          lrn_norm_region = 0;    // --lrn_norm_region=0|1 (0=across_ch, 1=within_ch)

    // Roll-specific
    std::string  roll_shift;             // --roll_shift=2:3:1

    // ShuffleChannels-specific
    int          shuffle_group = 2;      // --shuffle_group=N

    // Border-specific
    int          border_mode = 0;        // --border_mode=0|1|2|3 (const/edge/reflect/symmetric)
    float        border_value = 0.0f;    // --border_value=0.0

    // Reverse-specific
    int          reverse_mode = 0;       // --reverse_mode=0|1 (0=index, 1=mask)

    // ReverseSequence-specific
    int          seq_axis = 1;           // --seq_axis=N

    // OneHot-specific
    int          one_hot_depth = 0;      // --one_hot_depth=N
    float        one_hot_on_value = 1.0f;  // --one_hot_on_value=1.0
    float        one_hot_off_value = 0.0f; // --one_hot_off_value=0.0

    // GridSample-specific
    int          grid_mode = 0;          // --grid_mode=0|1|2 (bilinear/bicubic/nearest)
    int          grid_padding = 0;       // --grid_padding=0|1|2 (zeros/border/reflection)
    int          grid_align_corners = 0; // --grid_align_corners=0|1

    // ROI-specific
    int          roi_pooled_h = 7;       // --roi_pooled_h=N
    int          roi_pooled_w = 7;       // --roi_pooled_w=N
    float        roi_spatial_scale = 0.0625f;  // --roi_spatial_scale=0.0625
    int          roi_sampling_ratio = 2; // --roi_sampling_ratio=N

    // ExtractImagePatches-specific
    std::string  eip_sizes;              // --eip_sizes=3x3
    std::string  eip_strides;           // --eip_strides=1x1
    std::string  eip_rates;             // --eip_rates=1x1

    // RegionYolo-specific
    int          yolo_coords = 4;        // --yolo_coords=N
    int          yolo_classes = 80;      // --yolo_classes=N
    int          yolo_num = 3;           // --yolo_num=N
    int          yolo_do_softmax = 0;    // --yolo_do_softmax=0|1

    // DFT-specific
    int          dft_inverse = 0;        // --dft_inverse=0|1

    // Range-specific
    float        range_start = 0.0f;     // --range_start=0.0
    float        range_stop = 100.0f;    // --range_stop=100.0
    float        range_step = 1.0f;      // --range_step=1.0

    // Eye-specific
    int          eye_diagonal = 0;       // --eye_diagonal=N

    // NMS-specific
    int          nms_top_k = 100;        // --nms_top_k=N
    float        nms_iou_threshold = 0.5f;  // --nms_iou_threshold=0.5
    float        nms_score_threshold = 0.0f; // --nms_score_threshold=0.0

    // Batch file
    std::string  batch_file;            // --batch=file

    // Parsed common values
    std::vector<cldnn::data_types> data_types;
    impl_type    impl = impl_type::any;

    bool is_perf() const {
        return mode == bench_mode::perf || mode == bench_mode::fast_perf
            || mode == bench_mode::corr_perf || mode == bench_mode::run;
    }
    bool is_acc() const {
        return mode == bench_mode::acc || mode == bench_mode::corr_perf;
    }

    // Apply mode-specific iteration defaults (called after parse_common)
    void apply_mode_defaults() {
        switch (mode) {
            case bench_mode::fast_perf:
                warmup_iters = 1;
                perf_iters   = 3;
                break;
            case bench_mode::run:
                warmup_iters = 0;
                perf_iters   = 1;
                break;
            default:
                break;  // keep user-specified or CLI defaults
        }
    }

    // Generate __REPRO command string for benchdnn-style output
    std::string repro_str() const {
        std::string kflag = kernel_name;
        if (kernel_name == "fully_connected") kflag = "fc";
        else if (kernel_name == "convolution") kflag = "conv";
        else if (kernel_name == "scaled_dot_product_attention") kflag = "sdpa";
        else if (kernel_name == "deconvolution") kflag = "deconv";
        std::string s = "--" + kflag;
        if (!dt_str.empty()) s += " --dt=" + dt_str;
        if (!shapes_str.empty()) s += " --shapes=" + shapes_str;
        if (impl != impl_type::any) s += " --impl=" + impl2str(impl);
        if (!force_impl_str.empty()) s += " --force_impl=" + force_impl_str;
        if (!in_layouts_str.empty()) s += " --in_layouts=" + in_layouts_str;
        if (!out_layouts_str.empty()) s += " --out_layouts=" + out_layouts_str;
        if (!attr_scales_str.empty()) s += " --attr-scales=" + attr_scales_str;
        if (!attr_zero_points_str.empty()) s += " --attr-zero-points=" + attr_zero_points_str;
        if (!attr_post_ops_str.empty()) s += " --attr-post-ops=" + attr_post_ops_str;
        if (truncate) s += " --truncate";
        // Primitive-specific attributes (only emit non-default values)
        // -- Gemm --
        if (transpose_a != 0) s += " --transpose_a=" + std::to_string(transpose_a);
        if (transpose_b != 0) s += " --transpose_b=" + std::to_string(transpose_b);
        if (!gemm_order0.empty()) s += " --gemm_order0=" + gemm_order0;
        if (!gemm_order1.empty()) s += " --gemm_order1=" + gemm_order1;
        if (!gemm_order_out.empty()) s += " --gemm_order_out=" + gemm_order_out;
        // -- SDPA --
        if (is_causal >= 0) s += " --is_causal=" + std::to_string(is_causal);
        if (!order_q.empty()) s += " --order_q=" + order_q;
        if (!order_k.empty()) s += " --order_k=" + order_k;
        if (!order_v.empty()) s += " --order_v=" + order_v;
        if (!order_out.empty()) s += " --order_out=" + order_out;
        if (!scale_val.empty()) s += " --scale_val=" + scale_val;
        // -- Convolution --
        if (groups != 1) s += " --groups=" + std::to_string(groups);
        if (!strides.empty()) s += " --strides=" + strides;
        if (!dilations.empty()) s += " --dilations=" + dilations;
        if (!padding_begin.empty()) s += " --padding_begin=" + padding_begin;
        if (!padding_end.empty()) s += " --padding_end=" + padding_end;
        if (grouped_weights_shape != 0) s += " --grouped_weights_shape=" + std::to_string(grouped_weights_shape);
        // -- Pooling --
        if (pool_mode != 0) s += " --pool_mode=" + std::to_string(pool_mode);
        if (!pool_kernel.empty()) s += " --kernel=" + pool_kernel;
        if (!pool_strides.empty()) s += " --pool_strides=" + pool_strides;
        if (!pads_begin.empty()) s += " --pads_begin=" + pads_begin;
        if (!pads_end.empty()) s += " --pads_end=" + pads_end;
        if (rounding_type != 0) s += " --rounding_type=" + std::to_string(rounding_type);
        // -- Reduce --
        if (reduce_mode != 1) s += " --reduce_mode=" + std::to_string(reduce_mode);
        if (keep_dims != 1) s += " --keep_dims=" + std::to_string(keep_dims);
        if (!reduce_axes.empty()) s += " --reduce_axes=" + reduce_axes;
        // -- Softmax --
        if (axis != -1) s += " --axis=" + std::to_string(axis);
        // -- MVN --
        if (normalize_variance != 1) s += " --normalize_variance=" + std::to_string(normalize_variance);
        if (!epsilon.empty()) s += " --epsilon=" + epsilon;
        if (eps_inside_sqrt != 0) s += " --eps_inside_sqrt=" + std::to_string(eps_inside_sqrt);
        // -- Eltwise --
        if (eltwise_mode >= 0) s += " --eltwise_mode=" + std::to_string(eltwise_mode);
        if (pythondiv != 0) s += " --pythondiv=" + std::to_string(pythondiv);
        // -- SwiGLU --
        if (glu_type != 0) s += " --glu_type=" + std::to_string(glu_type);
        if (split_axis != -1) s += " --split_axis=" + std::to_string(split_axis);
        if (split_length != -1) s += " --split_length=" + std::to_string(split_length);
        if (gate_idx != 0) s += " --gate_idx=" + std::to_string(gate_idx);
        // -- Gather --
        if (gather_axis != 0) s += " --gather_axis=" + std::to_string(gather_axis);
        if (batch_dim != 0) s += " --batch_dim=" + std::to_string(batch_dim);
        if (support_neg_ind != 0) s += " --support_neg_ind=" + std::to_string(support_neg_ind);
        if (gather_rank != 0) s += " --gather_rank=" + std::to_string(gather_rank);
        // -- RoPE --
        if (head_cnt != 0) s += " --head_cnt=" + std::to_string(head_cnt);
        if (head_size != 0) s += " --head_size=" + std::to_string(head_size);
        if (rotary_ndims != 0) s += " --rotary_ndims=" + std::to_string(rotary_ndims);
        if (is_interleaved != 0) s += " --is_interleaved=" + std::to_string(is_interleaved);
        if (is_chatglm != 0) s += " --is_chatglm=" + std::to_string(is_chatglm);
        if (is_qwen != 0) s += " --is_qwen=" + std::to_string(is_qwen);
        if (input_trans0213 != 1) s += " --input_trans0213=" + std::to_string(input_trans0213);
        if (slice_start != 0) s += " --slice_start=" + std::to_string(slice_start);
        if (slice_stop != 0) s += " --slice_stop=" + std::to_string(slice_stop);
        // -- Crop --
        if (!offsets.empty()) s += " --offsets=" + offsets;
        // -- StridedSlice --
        if (!ss_begin.empty()) s += " --ss_begin=" + ss_begin;
        if (!ss_end.empty()) s += " --ss_end=" + ss_end;
        if (!ss_strides.empty()) s += " --ss_strides=" + ss_strides;
        if (!begin_mask.empty()) s += " --begin_mask=" + begin_mask;
        if (!end_mask.empty()) s += " --end_mask=" + end_mask;
        if (!shrink_axis_mask.empty()) s += " --shrink_axis_mask=" + shrink_axis_mask;
        if (!new_axis_mask.empty()) s += " --new_axis_mask=" + new_axis_mask;
        // -- Concatenation --
        if (concat_axis != 1) s += " --concat_axis=" + std::to_string(concat_axis);
        // -- Tile --
        if (!tile_repeats.empty()) s += " --tile_repeats=" + tile_repeats;
        // -- Normalize --
        if (across_spatial != 0) s += " --across_spatial=" + std::to_string(across_spatial);
        // -- GroupNormalization --
        if (num_groups != 1) s += " --num_groups=" + std::to_string(num_groups);
        // -- Quantize --
        if (levels != 0) s += " --levels=" + std::to_string(levels);
        // -- ScatterNDUpdate --
        if (indices_rank != 0) s += " --indices_rank=" + std::to_string(indices_rank);
        // -- Resample --
        if (!resample_sizes.empty()) s += " --resample_sizes=" + resample_sizes;
        if (resample_mode != 0) s += " --resample_mode=" + std::to_string(resample_mode);
        // -- Permute --
        if (!permute_order.empty()) s += " --permute_order=" + permute_order;
        // -- Broadcast --
        if (!broadcast_axes.empty()) s += " --broadcast_axes=" + broadcast_axes;
        if (!broadcast_target.empty()) s += " --broadcast_target=" + broadcast_target;
        // -- AdaptivePooling --
        if (adaptive_pool_mode != 0) s += " --adaptive_pool_mode=" + std::to_string(adaptive_pool_mode);
        if (!adaptive_pool_out.empty()) s += " --adaptive_pool_out=" + adaptive_pool_out;
        // -- ArgMaxMin (TopK) --
        if (topk_mode != 0) s += " --topk_mode=" + std::to_string(topk_mode);
        if (top_k != 1) s += " --top_k=" + std::to_string(top_k);
        // -- Col2Im --
        if (!col2im_output_shape.empty()) s += " --col2im_output_shape=" + col2im_output_shape;
        if (!col2im_kernel_shape.empty()) s += " --col2im_kernel_shape=" + col2im_kernel_shape;
        // -- DetectionOutput --
        if (det_num_classes != 21) s += " --det_num_classes=" + std::to_string(det_num_classes);
        if (det_keep_top_k != 200) s += " --det_keep_top_k=" + std::to_string(det_keep_top_k);
        if (det_top_k != -1) s += " --det_top_k=" + std::to_string(det_top_k);
        if (det_nms_threshold != 0.45f) { std::ostringstream oss; oss << det_nms_threshold; s += " --det_nms_threshold=" + oss.str(); }
        if (det_confidence_threshold != 0.01f) { std::ostringstream oss; oss << det_confidence_threshold; s += " --det_confidence_threshold=" + oss.str(); }
        if (det_code_type != 1) s += " --det_code_type=" + std::to_string(det_code_type);
        if (det_share_location != 1) s += " --det_share_location=" + std::to_string(det_share_location);
        if (det_background_label_id != 0) s += " --det_background_label_id=" + std::to_string(det_background_label_id);
        if (det_variance_encoded != 0) s += " --det_variance_encoded=" + std::to_string(det_variance_encoded);
        // -- CumSum --
        if (cum_exclusive != 0) s += " --cum_exclusive=" + std::to_string(cum_exclusive);
        if (cum_reverse != 0) s += " --cum_reverse=" + std::to_string(cum_reverse);
        // -- DepthToSpace / SpaceToDepth / ReorgYolo --
        if (block_size != 0) s += " --block_size=" + std::to_string(block_size);
        if (d2s_mode != 0) s += " --d2s_mode=" + std::to_string(d2s_mode);
        // -- GRN --
        if (grn_bias != 1e-6f) { std::ostringstream oss; oss << grn_bias; s += " --grn_bias=" + oss.str(); }
        // -- LRN --
        if (lrn_size != 5) s += " --lrn_size=" + std::to_string(lrn_size);
        if (lrn_k != 1.0f) { std::ostringstream oss; oss << lrn_k; s += " --lrn_k=" + oss.str(); }
        if (lrn_alpha != 0.0001f) { std::ostringstream oss; oss << lrn_alpha; s += " --lrn_alpha=" + oss.str(); }
        if (lrn_beta != 0.75f) { std::ostringstream oss; oss << lrn_beta; s += " --lrn_beta=" + oss.str(); }
        if (lrn_norm_region != 0) s += " --lrn_norm_region=" + std::to_string(lrn_norm_region);
        // -- Roll --
        if (!roll_shift.empty()) s += " --roll_shift=" + roll_shift;
        // -- ShuffleChannels --
        if (shuffle_group != 2) s += " --shuffle_group=" + std::to_string(shuffle_group);
        // -- Border --
        if (border_mode != 0) s += " --border_mode=" + std::to_string(border_mode);
        if (border_value != 0.0f) { std::ostringstream oss; oss << border_value; s += " --border_value=" + oss.str(); }
        // -- Reverse --
        if (reverse_mode != 0) s += " --reverse_mode=" + std::to_string(reverse_mode);
        // -- ReverseSequence --
        if (seq_axis != 1) s += " --seq_axis=" + std::to_string(seq_axis);
        // -- OneHot --
        if (one_hot_depth != 0) s += " --one_hot_depth=" + std::to_string(one_hot_depth);
        if (one_hot_on_value != 1.0f) { std::ostringstream oss; oss << one_hot_on_value; s += " --one_hot_on_value=" + oss.str(); }
        if (one_hot_off_value != 0.0f) { std::ostringstream oss; oss << one_hot_off_value; s += " --one_hot_off_value=" + oss.str(); }
        // -- GridSample --
        if (grid_mode != 0) s += " --grid_mode=" + std::to_string(grid_mode);
        if (grid_padding != 0) s += " --grid_padding=" + std::to_string(grid_padding);
        if (grid_align_corners != 0) s += " --grid_align_corners=" + std::to_string(grid_align_corners);
        // -- ROI --
        if (roi_pooled_h != 7) s += " --roi_pooled_h=" + std::to_string(roi_pooled_h);
        if (roi_pooled_w != 7) s += " --roi_pooled_w=" + std::to_string(roi_pooled_w);
        if (roi_spatial_scale != 0.0625f) { std::ostringstream oss; oss << roi_spatial_scale; s += " --roi_spatial_scale=" + oss.str(); }
        if (roi_sampling_ratio != 2) s += " --roi_sampling_ratio=" + std::to_string(roi_sampling_ratio);
        // -- ExtractImagePatches --
        if (!eip_sizes.empty()) s += " --eip_sizes=" + eip_sizes;
        if (!eip_strides.empty()) s += " --eip_strides=" + eip_strides;
        if (!eip_rates.empty()) s += " --eip_rates=" + eip_rates;
        // -- RegionYolo --
        if (yolo_coords != 4) s += " --yolo_coords=" + std::to_string(yolo_coords);
        if (yolo_classes != 80) s += " --yolo_classes=" + std::to_string(yolo_classes);
        if (yolo_num != 3) s += " --yolo_num=" + std::to_string(yolo_num);
        if (yolo_do_softmax != 0) s += " --yolo_do_softmax=" + std::to_string(yolo_do_softmax);
        // -- DFT --
        if (dft_inverse != 0) s += " --dft_inverse=" + std::to_string(dft_inverse);
        // -- Range --
        if (range_start != 0.0f) { std::ostringstream oss; oss << range_start; s += " --range_start=" + oss.str(); }
        if (range_stop != 100.0f) { std::ostringstream oss; oss << range_stop; s += " --range_stop=" + oss.str(); }
        if (range_step != 1.0f) { std::ostringstream oss; oss << range_step; s += " --range_step=" + oss.str(); }
        // -- Eye --
        if (eye_diagonal != 0) s += " --eye_diagonal=" + std::to_string(eye_diagonal);
        // -- NMS --
        if (nms_top_k != 100) s += " --nms_top_k=" + std::to_string(nms_top_k);
        if (nms_iou_threshold != 0.5f) { std::ostringstream oss; oss << nms_iou_threshold; s += " --nms_iou_threshold=" + oss.str(); }
        if (nms_score_threshold != 0.0f) { std::ostringstream oss; oss << nms_score_threshold; s += " --nms_score_threshold=" + oss.str(); }
        s += " --mode=" + mode2str(mode);
        s += " --device=" + std::to_string(device);
        return s;
    }

    void parse_common() {
        if (!dt_str.empty()) {
            data_types = parse_dt_list(dt_str);
        }
        if (!impl_str.empty()) {
            impl = str2impl(impl_str);
        }
        if (!mode_str.empty()) {
            mode = str2mode(mode_str);
        }
        apply_mode_defaults();
    }

    void print() const {
        std::cout << "bench_kernel config:" << std::endl;
        std::cout << "  kernel:     " << kernel_name << std::endl;
        std::cout << "  device:     " << device << std::endl;
        std::cout << "  dt:         " << dt_str << std::endl;
        std::cout << "  impl:       " << impl2str(impl) << std::endl;
        if (!force_impl_str.empty())
            std::cout << "  force_impl: " << force_impl_str << std::endl;
        if (!in_layouts_str.empty())
            std::cout << "  in_layouts: " << in_layouts_str << std::endl;
        if (!out_layouts_str.empty())
            std::cout << "  out_layouts:" << out_layouts_str << std::endl;
        std::cout << "  shapes:     " << shapes_str << std::endl;
        if (!attr_scales_str.empty())
            std::cout << "  scales:     " << attr_scales_str << std::endl;
        if (!attr_zero_points_str.empty())
            std::cout << "  zero_points:" << attr_zero_points_str << std::endl;
        if (!attr_post_ops_str.empty())
            std::cout << "  post_ops:   " << attr_post_ops_str << std::endl;
        std::cout << "  mode:       " << mode2str(mode) << std::endl;
        std::cout << "  warmup:     " << warmup_iters << std::endl;
        std::cout << "  perf_iters: " << perf_iters << std::endl;
        std::cout << std::endl;
    }
};

}  // namespace bench_kernel
