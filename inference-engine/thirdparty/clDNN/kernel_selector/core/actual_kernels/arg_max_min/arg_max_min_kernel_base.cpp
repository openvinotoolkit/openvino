/*
// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

KernelsData ArgMaxMinKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimatedTime) const {
    if (!Validate(params, options)) {
        return {};
    }

    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<arg_max_min_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = estimatedTime;

    return {kd};
}
}  // namespace kernel_selector