// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "quantize_kernel_scale_shift_opt.h"
#include "kernel_selector_utils.h"
#include <string>

static const size_t sub_group_size = 32;
static const size_t feature_size = 32;

namespace kernel_selector {
ParamsKey QuantizeKernelScaleShift::GetSupportedKey() const {
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
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    k.EnableDynamicShapesSupport();
    return k;
}

CommonDispatchData QuantizeKernelScaleShift::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatchData;

    auto output = params.outputs[0];

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16 || output.GetLayout() == DataLayout::b_fs_yx_fsv32 ||
        output.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        dispatchData.gws[0] = output.Z().v * output.Y().v * output.X().v;
        dispatchData.gws[1] = Align(output.Feature().v, sub_group_size);
        dispatchData.gws[2] = output.Batch().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;
    } else if (output.GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 || output.GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
               output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 || output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
               output.GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 || output.GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
               output.GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 || output.GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        dispatchData.gws[0] = output.Z().v * output.Y().v * output.X().v;
        dispatchData.gws[1] = Align(output.Feature().v, feature_size);
        dispatchData.gws[2] = Align(output.Batch().v, feature_size);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = feature_size;
        dispatchData.lws[2] = params.engineInfo.maxWorkGroupSize / feature_size;
    } else {
        dispatchData.gws = GetTensorFriendlyWorkGroups(output);
        auto out_layout = params.outputs[0].GetLayout();
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, out_layout, out_layout);
    }

    return dispatchData;
}

JitConstants QuantizeKernelScaleShift::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 || params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCKED_FORMAT", true));
        jit.AddConstant(MakeJitConstant("GWS_BATCH", 2));
        jit.AddConstant(MakeJitConstant("GWS_FEATURE", 1));
        jit.AddConstant(MakeJitConstant("GWS_YX", 0));
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    } else {
        auto tensor_jits = GetTensorFriendlyWorkGroupsJit(params.outputs[0]);
        jit.Merge(tensor_jits);
    }

    auto can_use_output_range = params.per_tensor_output_range && params.out_lo < params.out_hi;
    auto has_output_range_round = !(params.outputs[0].GetDType() == Datatype::INT8 || params.outputs[0].GetDType() == Datatype::UINT8);

    jit.AddConstant(MakeJitConstant("HAS_POST_SCALE", params.has_post_scale));
    jit.AddConstant(MakeJitConstant("HAS_POST_SHIFT", params.has_post_shift));
    jit.AddConstant(MakeJitConstant("HAS_PRE_SHIFT", params.has_pre_shift));
    jit.AddConstant(MakeJitConstant("HAS_CLAMP", params.has_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MIN_CLAMP", params.has_min_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MAX_CLAMP", params.has_max_clamp));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_RANGE", params.per_tensor_input_range));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_RANGE", params.per_tensor_output_range));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SCALE", params.per_tensor_input_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SHIFT", params.per_tensor_input_shift));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SCALE", params.per_tensor_output_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SHIFT", params.per_tensor_output_shift));
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

    return jit;
}

bool QuantizeKernelScaleShift::Validate(const Params& p) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        return false;

    return true;
}

KernelsPriority QuantizeKernelScaleShift::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
