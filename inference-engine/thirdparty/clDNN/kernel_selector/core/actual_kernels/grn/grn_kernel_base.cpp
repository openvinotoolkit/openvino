// Copyright (c) 2020 Intel Corporation
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


#include "grn_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants GRNKernelBase::GetJitConstants(const grn_params& params, GRNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("BIAS", params.bias));

    return jit;
}

GRNKernelBase::DispatchData GRNKernelBase::SetDefault(const grn_params& params) const {
    const auto& output = params.output;

    DispatchData kd;
    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    std::vector<size_t> global = { output.Batch().v, output.Y().v, output.X().v };
    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData GRNKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options,
                                                float estimated_time) const {
    assert(params.GetType() == KernelType::GRN);

    if (!Validate(params, options))
        return {};

    const grn_params& orgParams = static_cast<const grn_params&>(params);

    DispatchData runInfo;

    runInfo = SetDefault(orgParams);

    KernelData kd = KernelData::Default<grn_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, runInfo);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = estimated_time;

    return {kd};
}

}  // namespace kernel_selector
