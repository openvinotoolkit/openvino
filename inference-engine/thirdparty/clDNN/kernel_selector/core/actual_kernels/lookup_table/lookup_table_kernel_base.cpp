/*
// Copyright (c) 2018 Intel Corporation
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

#include "lookup_table_kernel_base.h"
#include <algorithm>

namespace kernel_selector {
bool LookUpTableKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::LOOKUP_TABLE || o.GetType() != KernelType::LOOKUP_TABLE) {
        return false;
    }

    return true;
}

JitConstants LookUpTableKernelBase::GetJitConstants(const lookup_table_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("VAL_NUM", params.numberOfValues),
        MakeJitConstant(toString(params.lookUpTableAxis) + "_AXIS", 1),
    });

    return jit;
}

LookUpTableKernelBase::DispatchData LookUpTableKernelBase::SetDefault(const lookup_table_params& params) const {
    DispatchData kd;

    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    // Determine global work sizes.
    kd.gws0 = params.inputIndices.X().v;
    kd.gws1 = params.inputIndices.Batch().v;  // B
    kd.gws2 = 1;

    kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (kd.gws0 % kd.lws0 != 0) {
        --kd.lws0;
    }
    kd.lws1 = 1;
    kd.lws2 = 1;

    return kd;
}

KernelsData LookUpTableKernelBase::GetCommonKernelsData(const Params& params,
                                                        const optional_params& options,
                                                        float estimatedTime) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lookup_table_params& orgParams = static_cast<const lookup_table_params&>(params);

    DispatchData runInfo = SetDefault(orgParams);

    KernelData kd = KernelData::Default<lookup_table_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

    kd.estimatedTime = estimatedTime;

    return {kd};
}
}  // namespace kernel_selector