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

#include "lookup_table_kernel_axis.h"
#include <algorithm>

namespace kernel_selector {
ParamsKey LookUpTableKernelAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableLookUpTableIndicesFormat(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableLookUpTableAxis(LookUpTableAxis::BATCH);
    k.EnableLookUpTableAxis(LookUpTableAxis::X);
    k.EnableLookUpTableAxis(LookUpTableAxis::Y);
    k.EnableLookUpTableAxis(LookUpTableAxis::FEATURE);
    k.EnableBatching();
    return k;
}

KernelsData LookUpTableKernelAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const lookup_table_params& orgParams = static_cast<const lookup_table_params&>(params);

    DispatchData runInfo;
    runInfo.fp16UnitUsed = orgParams.inputs[0].GetDType() == Datatype::F16;

    if (orgParams.lookUpTableAxis == LookUpTableAxis::BATCH) {
        runInfo.gws0 = orgParams.inputs[0].X().v;
        runInfo.gws1 = orgParams.inputs[0].Y().v;
        runInfo.gws2 = orgParams.inputs[0].Feature().v;
    } else if (orgParams.lookUpTableAxis == LookUpTableAxis::FEATURE) {
        runInfo.gws0 = orgParams.inputs[0].X().v;
        runInfo.gws1 = orgParams.inputs[0].Y().v;
        runInfo.gws2 = orgParams.inputs[0].Batch().v;
    } else if (orgParams.lookUpTableAxis == LookUpTableAxis::Y) {
        runInfo.gws0 = orgParams.inputs[0].X().v;
        runInfo.gws1 = orgParams.inputs[0].Feature().v;
        runInfo.gws2 = orgParams.inputs[0].Batch().v;
    } else if (orgParams.lookUpTableAxis == LookUpTableAxis::X) {
        runInfo.gws0 = orgParams.inputs[0].Y().v;
        runInfo.gws1 = orgParams.inputs[0].Feature().v;
        runInfo.gws2 = orgParams.inputs[0].Batch().v;
    }

    runInfo.lws0 = std::min(std::max(runInfo.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (runInfo.gws0 % runInfo.lws0 != 0) {
        --runInfo.lws0;
    }
    runInfo.lws1 = 1;
    runInfo.lws2 = 1;

    KernelData kd = KernelData::Default<lookup_table_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

    kd.estimatedTime = FORCE_PRIORITY_9;

    return {kd};
}
}  // namespace kernel_selector