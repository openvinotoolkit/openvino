// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_params : public base_params {
    quantize_params()
    : base_params(KernelType::QUANTIZE)
    , levels(0)
    , scale_shift_opt(false)
    , has_post_scale(true)
    , has_post_shift(true)
    , has_pre_shift(true)
    , has_clamp(true)
    , has_min_clamp(true)
    , has_max_clamp(true)
    , per_tensor_input_range(false)
    , per_tensor_input_scale(false)
    , per_tensor_input_shift(false)
    , per_tensor_output_range(false)
    , per_tensor_output_scale(false)
    , per_tensor_output_shift(false)
    , in_lo(0.0f)
    , in_hi(0.0f)
    , in_scale(0.0f)
    , in_shift(0.0f)
    , out_lo(0.0f)
    , out_hi(0.0f)
    , out_scale(0.0f)
    , out_shift(0.0f) { }

    int levels;
    bool scale_shift_opt;
    bool has_post_scale;
    bool has_post_shift;
    bool has_pre_shift;
    bool has_clamp;
    bool has_min_clamp;
    bool has_max_clamp;

    bool per_tensor_input_range;
    bool per_tensor_input_scale;
    bool per_tensor_input_shift;
    bool per_tensor_output_range;
    bool per_tensor_output_scale;
    bool per_tensor_output_shift;

    float in_lo;
    float in_hi;
    float in_scale;
    float in_shift;
    float out_lo;
    float out_hi;
    float out_scale;
    float out_shift;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        if (scale_shift_opt)
            k.EnableQuantizeScaleShiftOpt();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_fuse_params : fuse_params {
    quantize_fuse_params(bool scale_shift_opt,
                         bool has_post_scale,
                         bool has_post_shift,
                         bool has_pre_shift,
                         bool has_clamp,
                         bool has_min_clamp,
                         bool has_max_clamp,
                         bool per_tensor_input_range,
                         bool per_tensor_input_scale,
                         bool per_tensor_input_shift,
                         bool per_tensor_output_range,
                         bool per_tensor_output_scale,
                         bool per_tensor_output_shift,
                         float in_lo,
                         float in_hi,
                         float in_scale,
                         float in_shift,
                         float out_lo,
                         float out_hi,
                         float out_scale,
                         float out_shift)
    : fuse_params(KernelType::QUANTIZE)
    , scale_shift_opt(scale_shift_opt)
    , has_post_scale(has_post_scale)
    , has_post_shift(has_post_shift)
    , has_pre_shift(has_pre_shift)
    , has_clamp(has_clamp)
    , has_min_clamp(has_min_clamp)
    , has_max_clamp(has_max_clamp)
    , per_tensor_input_range(per_tensor_input_range)
    , per_tensor_input_scale(per_tensor_input_scale)
    , per_tensor_input_shift(per_tensor_input_shift)
    , per_tensor_output_range(per_tensor_output_range)
    , per_tensor_output_scale(per_tensor_output_scale)
    , per_tensor_output_shift(per_tensor_output_shift)
    , in_lo(in_lo)
    , in_hi(in_hi)
    , in_scale(in_scale)
    , in_shift(in_shift)
    , out_lo(out_lo)
    , out_hi(out_hi)
    , out_scale(out_scale)
    , out_shift(out_shift) {
        size_t index = 0;
        bool out_range_usage = per_tensor_output_range && out_lo < out_hi;
        if (!out_range_usage && has_clamp) {
            in_range_lo_idx = index++;
            in_range_hi_idx = index++;
        }
        if (!per_tensor_input_scale) {
            in_scale_idx = index++;
        }
        if (!per_tensor_input_shift && has_pre_shift) {
            in_shift_idx = index++;
        }
        if (!per_tensor_output_scale && has_post_scale) {
            out_scale_idx = index++;
        }
        if (!per_tensor_output_shift && has_post_shift) {
            out_shift_idx = index++;
        }
    }

    bool scale_shift_opt;
    bool has_post_scale;
    bool has_post_shift;
    bool has_pre_shift;
    bool has_clamp;
    bool has_min_clamp;
    bool has_max_clamp;

    bool per_tensor_input_range;
    bool per_tensor_input_scale;
    bool per_tensor_input_shift;
    bool per_tensor_output_range;
    bool per_tensor_output_scale;
    bool per_tensor_output_shift;

    float in_lo;
    float in_hi;
    float in_scale;
    float in_shift;
    float out_lo;
    float out_hi;
    float out_scale;
    float out_shift;

    size_t in_range_lo_idx = 0;
    size_t in_range_hi_idx = 0;
    size_t in_scale_idx = 0;
    size_t in_shift_idx = 0;
    size_t out_scale_idx = 0;
    size_t out_shift_idx = 0;
};

}  // namespace kernel_selector
