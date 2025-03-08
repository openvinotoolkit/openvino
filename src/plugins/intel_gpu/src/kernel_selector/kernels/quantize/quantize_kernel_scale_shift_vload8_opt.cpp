// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_scale_shift_vload8_opt.h"

#include <iostream>
#include <string>

#include "kernel_selector_utils.h"

static const size_t vec_size = 8;

namespace kernel_selector {
static inline int GetInnerFeatureBlockSize(const DataTensor&);
static inline int GetInnerBatchBlockSize(const DataTensor&);
static inline size_t CalculateTotalWorkItemCount(const quantize_params& params);

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
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    return k;
}

CommonDispatchData QuantizeKernelScaleShift_vload8::SetDefault(const quantize_params& params) const {
    CommonDispatchData dispatchData;
    dispatchData.gws[0] = CeilDiv(CalculateTotalWorkItemCount(params), vec_size);
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
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
    auto total_size = CalculateTotalWorkItemCount(params);
    if (total_size % vec_size) {
        // handle some leftovers
        jit.AddConstant(MakeJitConstant("LAST_ACCESSED_X", total_size - vec_size));
        jit.AddConstant(MakeJitConstant("LEFT_OVERS", total_size % vec_size));
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
    if (CalculateTotalWorkItemCount(params) < vec_size)
        return false;

    return true;
}

static inline size_t CalculateTotalWorkItemCount(const quantize_params& params) {
    if (!params.outputs[0].SimpleLayout()) {
        auto feature = Align(params.outputs[0].Feature().v, GetInnerFeatureBlockSize(params.outputs[0]));
        auto batch = Align(params.outputs[0].Batch().v, GetInnerBatchBlockSize(params.outputs[0]));
        size_t spatial = 0;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5)
            spatial = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
        else
            spatial = params.outputs[0].X().v * params.outputs[0].Y().v;

        return (feature * batch * spatial);
    } else {
        return params.outputs[0].LogicalSize();
    }
}

static inline int GetInnerBatchBlockSize(const DataTensor& tensor) {
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::bfyx:
    case DataLayout::byxf:
    case DataLayout::yxfb:
    case DataLayout::bfzyx:
    case DataLayout::b_fs_yx_fsv4:
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::fs_b_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::bfwzyx:
    case DataLayout::bfuwzyx:
    case DataLayout::bfvuwzyx:
        return 1;
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 16;
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
        return 32;
    default:
        OPENVINO_THROW("GetInnerBatchBlockSize : Unexpected format for quantize_vload8 opt kernel.");
    }

    return 1;
}

static inline int GetInnerFeatureBlockSize(const DataTensor& tensor) {
    auto layout = tensor.GetLayout();
    switch (layout) {
    case DataLayout::bfyx:
    case DataLayout::byxf:
    case DataLayout::yxfb:
    case DataLayout::bfzyx:
    case DataLayout::bfwzyx:
    case DataLayout::bfuwzyx:
    case DataLayout::bfvuwzyx:
        return 1;
    case DataLayout::b_fs_yx_fsv4:
        return 4;
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
    case DataLayout::bs_fs_yx_bsv32_fsv16:
    case DataLayout::bs_fs_yx_bsv16_fsv16:
    case DataLayout::bs_fs_zyx_bsv32_fsv16:
    case DataLayout::bs_fs_zyx_bsv16_fsv16:
        return 16;
    case DataLayout::b_fs_yx_fsv32:
    case DataLayout::fs_b_yx_fsv32:
    case DataLayout::b_fs_zyx_fsv32:
    case DataLayout::bs_fs_yx_bsv32_fsv32:
    case DataLayout::bs_fs_yx_bsv16_fsv32:
    case DataLayout::bs_fs_zyx_bsv32_fsv32:
    case DataLayout::bs_fs_zyx_bsv16_fsv32:
        return 32;
    default:
        OPENVINO_THROW("GetInnerFeatureBlockSize : Unexpected format for quantize_vload8 opt kernel.");
    }

    return 1;
}

KernelsPriority QuantizeKernelScaleShift_vload8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
