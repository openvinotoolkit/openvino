// Copyright (c) 2019 Intel Corporation
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

    DispatchData kd;

    kd.fp16UnitUsed = input.GetDType() == Datatype::F16;

    std::vector<size_t> global{input.Batch().v, input.Feature().v, input.Y().v * input.X().v};
    if (params.output.GetDims().size() == 5) {
        global[0] = input.Batch().v;
        global[1] = input.Feature().v * input.Z().v;
        global[2] = input.Y().v * input.X().v;
    }
    const auto& local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData OneHotKernelBase::GetCommonKernelsData(const Params& params,
                                                   const optional_params& options,
                                                   float estimated_time) const {
    assert(params.GetType() == KernelType::ONE_HOT);

    const auto& prim_params =
        static_cast<const one_hot_params&>(params);

    auto run_info = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<one_hot_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, run_info, params.engineInfo, kernelName, jit, entry_point);
    k_data.estimatedTime = estimated_time;

    return {k_data};
}
}  // namespace kernel_selector
