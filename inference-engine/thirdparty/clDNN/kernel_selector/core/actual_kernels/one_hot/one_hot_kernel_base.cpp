// Copyright (c) 2019-2020 Intel Corporation
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

#include "one_hot_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants OneHotKernelBase::GetJitConstants(const one_hot_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
         MakeJitConstant("ONE_HOT_AXIS", params.one_hot_axis),
         MakeJitConstant("ONE_HOT_LIMIT", params.one_hot_limit),
         MakeJitConstant("ON_VALUE", params.on_value),
         MakeJitConstant("OFF_VALUE", params.off_value)
    });

    return jit;
}

OneHotKernelBase::DispatchData OneHotKernelBase::SetDefault(const one_hot_params& params) {
    const auto& input = params.inputs[0];

    DispatchData dispatchData;
    if (params.output.GetDims().size() == 5) {
        dispatchData.gws = { input.Batch().v, input.Feature().v * input.Z().v, input.Y().v * input.X().v };
    } else {
        dispatchData.gws = { input.Batch().v, input.Feature().v, input.Y().v * input.X().v };
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData OneHotKernelBase::GetCommonKernelsData(const Params& params,
                                                   const optional_params& options,
                                                   float estimated_time) const {
    assert(params.GetType() == KernelType::ONE_HOT);

    const auto& prim_params =
        static_cast<const one_hot_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<one_hot_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point);
    k_data.estimatedTime = estimated_time;

    return {k_data};
}
}  // namespace kernel_selector
