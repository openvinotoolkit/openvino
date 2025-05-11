// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_to_yxfb_batched.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernel_to_yxfb_batched::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ReorderKernel_to_yxfb_batched::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_reqd_subgroup_size();

    return k;
}

bool ReorderKernel_to_yxfb_batched::Validate(const Params& params) const {
    if (!ReorderKernelBase::Validate(params)) {
        return false;
    }

    if (!IsSIMDSizeSupported(params.engineInfo, 8))
        return false;

    const reorder_params& r_params = static_cast<const reorder_params&>(params);

    const auto& output = r_params.outputs[0];
    // output cannot have padding for this implementation
    if (output.X().pad.Total() != 0 || output.Y().pad.Total() != 0 || output.Feature().pad.Total() != 0 ||
        output.Batch().pad.Total() != 0) {
        return false;
    }

    if ((r_params.inputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 || r_params.inputs[0].GetLayout() == DataLayout::bs_fs_zyx_bsv16_fsv16) &&
        r_params.inputs[0].Z().v != 1)
        return false;

    return true;
}

JitConstants ReorderKernel_to_yxfb_batched::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    KernelData kd = KernelData::Default<reorder_params>(params);
    reorder_params& newParams = *static_cast<reorder_params*>(kd.params.get());

    const auto& input = newParams.inputs[0];
    jit.AddConstant(MakeJitConstant("ELEMENTS_COUNT", input.LogicalSize()));

    return jit;
}

ReorderKernelBase::DispatchData ReorderKernel_to_yxfb_batched::SetDefault(const reorder_params& params) const {
    DispatchData dispatchData;

    const auto& input = params.inputs[0];

    unsigned int gws = (unsigned int)input.LogicalSize();

    dispatchData.gws[0] = Align(gws, 8 * input.Batch().v) / input.Batch().v;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 8;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData ReorderKernel_to_yxfb_batched::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernel_to_yxfb_batched::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const reorder_params&>(params);

    return p.inputs[0].Batch().v == 1 ? DONT_USE_IF_HAVE_SOMETHING_ELSE : FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
