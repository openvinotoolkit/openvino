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

#include "scale_grad_weights_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector
{
    JitConstants ScaleGradWeightsKernelBase::GetJitConstants(const scale_grad_weights_params& params) const
    {
        JitConstants jit = training_kernel_base::GetJitConstants(params);
        
        return jit;
    }

    ScaleGradWeightsKernelBase::DispatchData ScaleGradWeightsKernelBase::SetDefault(const scale_grad_weights_params& params) const
    {
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        kd.gws0 = params.inputs[0].Batch().v;
        kd.gws1 = params.inputs[0].Feature().v;
        kd.gws2 = 1;

        kd.lws0 = params.inputs[0].Batch().v;
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }

    KernelsData ScaleGradWeightsKernelBase::GetKernelsData(const Params& params, const optional_params& options) const
    {
        assert(params.GetType() == KernelType::SCALE_GRAD_WEIGHTS);

        const scale_grad_weights_params& orgParams = static_cast<const scale_grad_weights_params&>(params);

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<scale_grad_weights_params>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point, ROUND_ROBIN, true, !orgParams.bias.empty(), 2);

        if (orgParams.use_momentum)
        {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::PREV_WEIGHTS_GRADIENT, 0 });
            if (!orgParams.bias.empty())
                kernel.arguments.push_back({ ArgumentDescriptor::Types::PREV_BIAS_GRADIENT, 0 });
        }
        kernel.arguments.push_back({ ArgumentDescriptor::Types::LEARNING_RATE, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}