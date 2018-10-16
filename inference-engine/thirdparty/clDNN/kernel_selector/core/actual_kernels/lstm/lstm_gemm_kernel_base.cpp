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

#include "lstm_gemm_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"

namespace kernel_selector
{
    JitConstants LSTMGemmKernelBase::GetJitConstants(const lstm_gemm_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);
        const auto& weights = params.weights;
        const auto& recurrent = params.recurrent;
        const auto& hidden = params.hidden;
        const auto& bias = params.bias;
        if (params.hasBias) {
            jit.AddConstants({ MakeJitConstant("BIAS", bias), MakeJitConstant("BIAS_TERM", true) });
        }
        if (params.hasHidden) {
            jit.AddConstants({ MakeJitConstant("HIDDEN", hidden), MakeJitConstant("HIDDEN_TERM", true) , MakeJitConstant("RECURRENT", recurrent) });
        }

        jit.AddConstants({ MakeJitConstant("WEIGHTS", weights)});

        return jit;
    }

    KernelsData LSTMGemmKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const
    {
        if (!Validate(params,  options))
        {
            return{};
        }

        const lstm_gemm_params& orgParams = static_cast<const lstm_gemm_params&>(params);

        KernelData kd = KernelData::Default<lstm_gemm_params>(params, orgParams.inputs.size());

        float effiency = FORCE_PRIORITY_1;
        const auto& input = orgParams.inputs[0];

        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;
        auto out = newParams.output;
        //TODO: reorder weights if needed
        auto& kernel = kd.kernels[0];
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.workGroups.global = { out.X().v, out.Batch().v, 1 };
        kernel.kernelString = GetKernelString(kernelName, jit, entryPoint);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        kernel.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        if (orgParams.hasHidden) {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::HIDDEN, 0 });
            kernel.arguments.push_back({ ArgumentDescriptor::Types::RECURRENT, 0 });
        }
        if (orgParams.hasBias) {
            kernel.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        }

        kd.estimatedTime = effiency;

        return{ kd };
    }
}