// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

bool PermuteKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::PERMUTE || o.GetType() != KernelType::PERMUTE) {
        return false;
    }
    const permute_params& params = static_cast<const permute_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op)) {
            return false;
        }
    }
    return true;
}

JitConstants PermuteKernelBase::GetJitConstants(const permute_params& params, const CommonDispatchData&) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    return jit;
}

KernelsData PermuteKernelBase::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<permute_params>(params);
    permute_params& newParams = *static_cast<permute_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    std::pair<std::string, std::string> jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 1, GetFusedPrimitiveInputsCount(params));

    return {kd};
}
}  // namespace kernel_selector
