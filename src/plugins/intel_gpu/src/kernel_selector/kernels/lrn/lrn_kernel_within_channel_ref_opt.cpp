// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_kernel_within_channel_ref_opt.h"

namespace kernel_selector {
ParamsKey LRNKernelWithinChannelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableLRNMode(LRNMode::WITHIN_CHANNEL);
    k.EnableLRNKernelDividerMode(KernelDividerMode::FIXED);
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData LRNKernelWithinChannelOpt::SetDefault(const lrn_params& params) const {
    CommonDispatchData dispatchData = LRNKernelBase::SetDefault(params);
    const auto totalSize = params.inputs[0].LogicalSize();
    const unsigned work_group_size = (totalSize < 128) ? 32 : 128;

    dispatchData.gws[0] = Align(params.inputs[0].LogicalSize(), work_group_size);
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = work_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

bool LRNKernelWithinChannelOpt::Validate(const Params& p) const {
    if (!LRNKernelBase::Validate(p)) {
        return false;
    }
    return true;
}

JitConstants LRNKernelWithinChannelOpt::GetJitConstants(const lrn_params& params, const LRNKernelWithinChannelOpt::Parent::DispatchData& dispatchData) const {
    const auto& input_dt = params.inputs[0].GetDType();
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"", {"batch_id", "feature_id", "y", "x"}, "lrn_result", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

KernelsData LRNKernelWithinChannelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority LRNKernelWithinChannelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
