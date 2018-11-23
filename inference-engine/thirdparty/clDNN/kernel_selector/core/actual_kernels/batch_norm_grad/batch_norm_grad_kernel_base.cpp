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

#include "batch_norm_grad_kernel_base.h"

namespace kernel_selector
{
    bool BatchNormGradKernelBase::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::BATCH_NORM_GRAD ||
            o.GetType() != KernelType::BATCH_NORM_GRAD)
        {
            return false;
        }

        return true;
    }

    JitConstants BatchNormGradKernelBase::GetJitConstants(const batch_norm_grad_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);
        return jit;
    }

    BatchNormGradKernelBase::DispatchData BatchNormGradKernelBase::SetDefault(const batch_norm_grad_params& params) const
    {
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        kd.gws0 = params.inputs[0].Batch().v;
        kd.gws1 = params.inputs[0].Feature().v;
        kd.gws2 = 1;

        kd.lws0 = params.inputs[0].Batch().v;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData BatchNormGradKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimatedTime) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const batch_norm_grad_params& orgParams = static_cast<const batch_norm_grad_params&>(params);

        DispatchData runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<batch_norm_grad_params>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3);

        kd.estimatedTime = estimatedTime;

        return{ kd };
    }
}