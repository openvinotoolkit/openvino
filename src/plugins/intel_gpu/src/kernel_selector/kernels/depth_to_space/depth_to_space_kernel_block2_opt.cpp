// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_kernel_block2_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey DepthToSpaceKernelBlock2Opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    return k;
}

bool DepthToSpaceKernelBlock2Opt::Validate(const Params& p) const {
    if (!DepthToSpaceKernelBase::Validate(p))
        return false;

    const auto& params = static_cast<const depth_to_space_params&>(p);

    if ((params.block_size != 2) || (params.inputs[0].X().v % 2 != 0))
        return false;

    if (params.mode != DepthToSpaceMode::BLOCKS_FIRST)
        return false;

    return true;
}

CommonDispatchData DepthToSpaceKernelBlock2Opt::SetDefault(const depth_to_space_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = { Align(params.inputs[0].X().v / 2, 16),
                         params.inputs[0].Y().v,
                         1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants DepthToSpaceKernelBlock2Opt::GetJitConstants(const depth_to_space_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("IN_WIDTH", params.inputs[0].X().v / 2));

    return jit;
}

KernelsData DepthToSpaceKernelBlock2Opt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority DepthToSpaceKernelBlock2Opt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
