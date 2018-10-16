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

#include "arg_max_min_kernel_opt.h"

namespace kernel_selector
{
    ParamsKey ArgMaxMinKernelOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableArgMaxMinAxis(ArgMaxMinAxis::XYF);
        k.EnableDifferentTypes();
        return k;
    }

    KernelsData ArgMaxMinKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }
      
        const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

        int topK = orgParams.topK;
        long size = (long)(orgParams.inputs[0].X().v * orgParams.inputs[0].Y().v * orgParams.inputs[0].Feature().v) / 8;
        long outSize = size/16 * topK;
        int kernelAmount = 1;
        for (; outSize > 128; outSize = (long)((outSize / 128 + 1) * topK))
        {
            kernelAmount++;
        }
        KernelData kd = KernelData::Default<arg_max_min_params>(params, kernelAmount);
        for (int i = 0; i < kernelAmount; i++)
        {
            DataTensor input;
            if (i == 0)
                input = orgParams.inputs[0];
            else
                input = orgParams.output;

            auto newParams = orgParams;
            newParams.inputs.resize(1);
            newParams.inputs[0] = input;

            auto& kernel = kd.kernels[i];
            DispatchData runInfo = SetDefault(newParams);
            auto cldnnJit = GetJitConstants(newParams);
            auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, options);
            auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

            runInfo.fp16UnitUsed = orgParams.inputs[0].GetDType() == Datatype::F16;

            runInfo.gws0 = Align(size, 16);
            runInfo.gws1 = orgParams.inputs[0].Batch().v;                  // B
            runInfo.gws2 = 1;

            runInfo.lws0 = 16;
            runInfo.lws1 = 1;
            runInfo.lws2 = 1;

            FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);
            size = (size / 128 + 1) * topK;
        }

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}