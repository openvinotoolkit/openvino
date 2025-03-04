// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_scale_shift_vload8_opt.h"

#include <iostream>
#include <string>

#include "kernel_selector_utils.h"

static const size_t vec_size = 8;

namespace kernel_selector {
ParamsKey QuantizeKernelScaleShift_vload8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    return k;
}

auto parse_block_size = [](int index, DataLayout dl) {
    std::string format_str = toString(dl);
    auto get_block_size = [&] (std::string substr) {
        auto start_pos = format_str.find(substr);
        if (start_pos != std::string::npos) {
            auto end_pos = format_str.find("_", start_pos);
            auto sub_string = format_str.substr(start_pos + strlen(substr.c_str()) , end_pos);
            return std::atoi(sub_string.c_str());
        }
        return 1;
    };
    return index == 0 ? get_block_size("BSV") : (index == 1 ? get_block_size("FSV") : 1);
};

auto get_total_size = [](const quantize_params& params) {
    const auto input = params.inputs[0];
    size_t totalSize = input.LogicalSize();
    auto feature_block_size = parse_block_size(1, input.GetLayout());
    auto feature_division = feature_block_size > 1 ? (input.Feature().v ? input.Feature().v : 1) : 1;
    auto feature_align_multiplexer = feature_block_size > 1 ? Align(input.Feature().v, feature_block_size) : 1;
    auto batch_block_size = parse_block_size(0, input.GetLayout());
    auto batch_divsion = batch_block_size > 1 ? (input.Batch().v ? input.Batch().v : 1) : 1;
    auto batch_align_multiplexer = batch_block_size > 1 ? Align(input.Batch().v, batch_block_size) : 1;
    return (totalSize / (feature_division * batch_divsion)) * feature_align_multiplexer * batch_align_multiplexer;
};

CommonDispatchData QuantizeKernelScaleShift_vload8::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatchData;
    if (true) {
        dispatchData.gws[0] = CeilDiv(get_total_size(params), vec_size);
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

JitConstants QuantizeKernelScaleShift_vload8::GetJitConstants(const quantize_params& params,
                                                              const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    auto can_use_output_range = params.per_tensor_output_range && params.out_lo < params.out_hi;
    auto has_output_range_round =
        !(params.outputs[0].GetDType() == Datatype::INT8 || params.outputs[0].GetDType() == Datatype::UINT8);

    jit.AddConstant(MakeJitConstant("HAS_POST_SCALE", params.has_post_scale));
    jit.AddConstant(MakeJitConstant("HAS_POST_SHIFT", params.has_post_shift));
    jit.AddConstant(MakeJitConstant("HAS_PRE_SHIFT", params.has_pre_shift));
    jit.AddConstant(MakeJitConstant("HAS_CLAMP", params.has_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MIN_CLAMP", params.has_min_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MAX_CLAMP", params.has_max_clamp));
    jit.AddConstant(MakeJitConstant("IN_LO_VAL", params.in_lo));
    jit.AddConstant(MakeJitConstant("IN_HI_VAL", params.in_hi));
    jit.AddConstant(MakeJitConstant("OUT_LO_VAL", params.out_lo));
    jit.AddConstant(MakeJitConstant("OUT_HI_VAL", params.out_hi));
    jit.AddConstant(MakeJitConstant("IN_SCALE_VAL", params.in_scale));
    jit.AddConstant(MakeJitConstant("IN_SHIFT_VAL", params.in_shift));
    jit.AddConstant(MakeJitConstant("OUT_SCALE_VAL", params.out_scale));
    jit.AddConstant(MakeJitConstant("OUT_SHIFT_VAL", params.out_shift));
    jit.AddConstant(MakeJitConstant("CAN_USE_OUTPUT_RANGE", can_use_output_range));
    jit.AddConstant(MakeJitConstant("HAS_OUTPUT_RANGE_ROUND", has_output_range_round));
    if (get_total_size(params) % vec_size) {
        // handle some leftovers
        jit.AddConstant(MakeJitConstant("LAST_ACCESSED_X", get_total_size(params) - vec_size));
        jit.AddConstant(MakeJitConstant("LEFT_OVERS", get_total_size(params) % vec_size));
    }
    return jit;
}

bool QuantizeKernelScaleShift_vload8::Validate(const Params& p) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        return false;

    // this kernel is opt for per tensor quantization params for now
    if (!params.per_tensor_input_range || !params.per_tensor_output_range || !params.per_tensor_input_scale ||
        !params.per_tensor_output_scale || !params.per_tensor_output_shift ||
        (params.has_pre_shift && !params.per_tensor_input_shift) ||
        params.outputs[0].GetLayout() != params.inputs[0].GetLayout() ||
        params.inputs[1].GetDType() != params.inputs[3].GetDType())
        return false;

    // for blocked format, if extra padding exist in a block, will be opt in a seprate kernel
    if (!params.inputs[0].SimpleLayout()) {
        const auto input_layout = params.inputs[0].GetLayout();
        const auto batch_size = params.inputs[0].Batch().v;
        const auto feature_size = params.inputs[0].Feature().v;
        if (!params.inputs[0].SimpleLayout())
            if (((input_layout == DataLayout::b_fs_yx_fsv16 || input_layout == DataLayout::b_fs_zyx_fsv16) &&
                 feature_size % 16 != 0) ||
                ((input_layout == DataLayout::b_fs_yx_fsv32 || input_layout == DataLayout::b_fs_zyx_fsv32) &&
                 feature_size % 32 != 0) ||
                (input_layout == DataLayout::b_fs_yx_fsv4 && feature_size % 8 != 0) ||
                input_layout == DataLayout::fs_b_yx_fsv32 ||
                ((input_layout == DataLayout::bs_fs_yx_bsv32_fsv16 ||
                  input_layout == DataLayout::bs_fs_zyx_bsv32_fsv16) &&
                 (feature_size % 16 != 0 || batch_size % 32 != 0)) ||
                ((input_layout == DataLayout::bs_fs_yx_bsv32_fsv32 ||
                  input_layout == DataLayout::bs_fs_zyx_bsv32_fsv32) &&
                 (feature_size % 32 != 0 || batch_size % 32 != 0)) ||
                ((input_layout == DataLayout::bs_fs_yx_bsv16_fsv16 ||
                  input_layout == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
                 (feature_size % 16 != 0 || batch_size % 16 != 0)) ||
                ((input_layout == DataLayout::bs_fs_yx_bsv16_fsv32 ||
                  input_layout == DataLayout::bs_fs_zyx_bsv16_fsv32) &&
                 (feature_size % 32 != 0 || batch_size % 16 != 0)))
                return false;
    }
    if (get_total_size(params) < vec_size)
        return false;
    return true;
}

KernelsPriority QuantizeKernelScaleShift_vload8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
