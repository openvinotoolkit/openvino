// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool DepthToSpaceKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::DEPTH_TO_SPACE) {
        return false;
    }

    const depth_to_space_params& params = static_cast<const depth_to_space_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

JitConstants DepthToSpaceKernelBase::GetJitConstants(const depth_to_space_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", params.block_size));
    if (params.mode == DepthToSpaceMode::BLOCKS_FIRST) {
        jit.AddConstant(MakeJitConstant("BLOCKS_FIRST", 1));
    } else {
        jit.AddConstant(MakeJitConstant("DEPTH_FIRST", 1));
    }

    return jit;
}

KernelsData DepthToSpaceKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<depth_to_space_params>(params);
    depth_to_space_params& newParams = *static_cast<depth_to_space_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    return { kd };
}
}  // namespace kernel_selector
