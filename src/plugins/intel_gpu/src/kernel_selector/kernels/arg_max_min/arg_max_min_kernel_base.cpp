// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_base.h"

namespace kernel_selector {
bool ArgMaxMinKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::ARG_MAX_MIN ||
        o.GetType() != KernelType::ARG_MAX_MIN) {
        return false;
    }

    return true;
}

JitConstants ArgMaxMinKernelBase::GetJitConstants(const arg_max_min_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("TOP_K", params.topK),
                      MakeJitConstant(toString(params.argMaxMinAxis) + "_AXIS", 1),
                      params.argMaxMinOut == ArgMaxMinOut::MAX ? MakeJitConstant("MAX_OUT", 1) : MakeJitConstant("MIN_OUT", 1)});

    return jit;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelBase::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws = { 128, params.inputs[0].Batch().v, 1 };
    dispatchData.lws = { 128, 1, 1 };

    return dispatchData;
}

KernelsData ArgMaxMinKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<arg_max_min_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    return {kd};
}
}  // namespace kernel_selector
