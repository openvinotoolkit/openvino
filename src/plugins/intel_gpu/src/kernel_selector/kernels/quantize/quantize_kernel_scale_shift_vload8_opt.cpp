// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_scale_shift_vload8_opt.h"

#include <iostream>
#include <string>

#include "kernel_selector_utils.h"

static const size_t sub_group_size = 32;
static const size_t feature_size = 32;

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
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    k.EnableDynamicShapesSupport();
    return k;
}

CommonDispatchData QuantizeKernelScaleShift_vload8::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatchData;
    // need special handle for blocked format??
    if (true) {
        dispatchData.gws[0] = std::max(params.outputs[0].LogicalSize() / 8, (size_t)1);
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes({dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2]},
                                                     params.engineInfo);
    return dispatchData;
}

JitConstants QuantizeKernelScaleShift_vload8::GetJitConstants(const quantize_params& params,
                                                              const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv16 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
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

    return jit;
}

bool QuantizeKernelScaleShift_vload8::Validate(const Params& p) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        return false;

    // this kernel is opt for per tensor quantization params for now
    if (!params.per_tensor_input_range || !params.per_tensor_output_range || !params.per_tensor_input_scale ||
        !params.per_tensor_output_scale || !params.per_tensor_output_shift ||
        (params.has_pre_shift && !params.per_tensor_input_shift))
        return false;
    // TBD, do we really need the strick block_size checking to support blocked foramt?
    for (size_t i = 0; i < params.inputs.size(); i++) {
        const auto input_layout = params.inputs[i].GetLayout();
        const auto batch_size = params.inputs[i].Batch().v;
        const auto feature_size = params.inputs[i].Feature().v;
        if ((input_layout == DataLayout::b_fs_yx_fsv16 && feature_size % 16 != 0) ||
            (input_layout == DataLayout::b_fs_yx_fsv32 && feature_size % 32 != 0) ||
            (input_layout == DataLayout::b_fs_zyx_fsv16 && feature_size % 16 != 0) ||
            (input_layout == DataLayout::b_fs_yx_fsv4 && feature_size % 8 != 0) ||
            input_layout == DataLayout::fs_b_yx_fsv32 ||
            (input_layout == DataLayout::bs_fs_yx_bsv32_fsv16 && (feature_size % 16 != 0 || batch_size % 32 != 0)) ||
            (input_layout == DataLayout::bs_fs_yx_bsv32_fsv32 && (feature_size % 32 != 0 || batch_size % 32 != 0)))
            return false;
    }
    if ((params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 && params.outputs[0].Feature().v % 16 != 0) ||
        (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 && params.outputs[0].Feature().v % 32 != 0) ||
        (params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 && params.outputs[0].Feature().v % 16 != 0) ||
        (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv4 && params.outputs[0].Feature().v % 8 != 0) ||
        params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32 ||
        (params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 &&
         (params.outputs[0].Feature().v % 16 != 0 || params.outputs[0].Batch().v % 32 != 0)) ||
        (params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 &&
         (params.outputs[0].Feature().v % 32 != 0 || params.outputs[0].Batch().v % 32 != 0)))
        return false;
    // TBD maybe need more stric check?
    return true;
}

KernelsData QuantizeKernelScaleShift_vload8::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::QUANTIZE);

    KernelData kd = KernelData::Default<quantize_params>(params);
    quantize_params& nparams = *static_cast<quantize_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(nparams);
    auto entry_point = GetEntryPoint(kernelName, nparams.layerID, params);
    auto cldnn_jit = GetJitConstants(nparams, dispatchData);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];

    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments =
        GetArgsDesc(static_cast<int>(nparams.inputs.size()), false, false, 0, 1, nparams.has_dynamic_tensors());

    return {kd};
}

KernelsPriority QuantizeKernelScaleShift_vload8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
