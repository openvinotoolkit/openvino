// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_fast_b1.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernelFastBatch1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableInputLayout(DataLayout::bs_f_bsv8__af8);
    k.EnableInputLayout(DataLayout::bs_f_bsv16__af8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bs_f_bsv8__af8);
    k.EnableOutputLayout(DataLayout::bs_f_bsv16__af8);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

bool ReorderKernelFastBatch1::Validate(const Params& p) const {
    if (!ReorderKernelBase::Validate(p)) {
        return false;
    }

    const reorder_params& params = static_cast<const reorder_params&>(p);

    if (params.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32)
        return false;

    if (params.inputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32)
        return false;

    return true;
}

JitConstants ReorderKernelFastBatch1::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));

    const auto& output = newParams.outputs[0];

    if (input.GetLayout() == output.GetLayout() && input.SameDimsSizes(output) &&
        !input.PitchesDifferFromLogicalDims() && !output.PitchesDifferFromLogicalDims() &&
        input.GetDType() != output.GetDType() && !params.has_padded_output &&
        params.mode == MeanSubtractMode::NONE) {
        jit.AddConstant(MakeJitConstant("CHANGE_DATA_TYPE_ONLY", 1));
    }

    return jit;
}

ReorderKernelFastBatch1::DispatchData ReorderKernelFastBatch1::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.outputs[0];

    unsigned int gws = (unsigned int)output.LogicalSize();

    dispatchData.gws[0] = Align(gws, 32);
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 32;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderKernelFastBatch1::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernelFastBatch1::GetKernelsPriority(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    const auto& input = orgParams.inputs[0];
    const auto& output = orgParams.outputs[0];

    return input.Batch().v == 1 && output.Batch().v == 1 ? FORCE_PRIORITY_6 : DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
