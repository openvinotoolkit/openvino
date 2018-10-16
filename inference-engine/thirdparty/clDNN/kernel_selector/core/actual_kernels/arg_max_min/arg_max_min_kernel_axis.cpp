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

#include "arg_max_min_kernel_axis.h"

namespace kernel_selector
{
    ParamsKey ArgMaxMinKernelAxis::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::F32);  //We support only f32, look into arg_max_min.hpp for more informations.
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
        k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
        k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
        k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
        k.EnableDifferentTypes();
        k.EnableBatching();
        return k;
    }

    KernelsData ArgMaxMinKernelAxis::GetKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

        DispatchData runInfo;
        runInfo.fp16UnitUsed = orgParams.inputs[0].GetDType() == Datatype::F16;

        runInfo.gws0 = 128;
        if (orgParams.argMaxMinAxis == ArgMaxMinAxis::BATCH) {
            runInfo.gws1 = orgParams.inputs[0].X().v;
            runInfo.gws2 = orgParams.inputs[0].Feature().v * orgParams.inputs[0].Y().v; 
        }
        else if (orgParams.argMaxMinAxis == ArgMaxMinAxis::FEATURE) {
            runInfo.gws1 = orgParams.inputs[0].X().v;
            runInfo.gws2 = orgParams.inputs[0].Batch().v * orgParams.inputs[0].Y().v;
        }
        else if (orgParams.argMaxMinAxis == ArgMaxMinAxis::Y) {
            runInfo.gws1 = orgParams.inputs[0].X().v;
            runInfo.gws2 = orgParams.inputs[0].Feature().v * orgParams.inputs[0].Batch().v;
        }
        else if (orgParams.argMaxMinAxis == ArgMaxMinAxis::X) {
            runInfo.gws1 = orgParams.inputs[0].Y().v;
            runInfo.gws2 = orgParams.inputs[0].Feature().v * orgParams.inputs[0].Batch().v;
        }

        runInfo.lws0 = 128;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        KernelData kd = KernelData::Default<arg_max_min_params>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}