/*
// Copyright (c) 2016 Intel Corporation
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

#include "normalize_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector 
{
    JitConstants NormalizeKernelBase::GetJitConstants(const normalize_params& np) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(np);

        jit.AddConstants({
            MakeJitConstant("SCALE_TABLE",          np.scaleTable),
            MakeJitConstant("EPSILON",              np.epsilon),
            MakeJitConstant(toString(np.normMode),  ""),
            MakeJitConstant("THRESHOLD",            0.0001f),
        });

        return jit;
    }

    NormalizeKernelBase::DispatchData NormalizeKernelBase::SetDefault(const normalize_params& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        std::vector<size_t> global(3);

        if (params.normMode == NormalizeMode::WITHIN_SPATIAL)
        {
            global = { output.X().v, output.Y().v, output.Batch().v };
        }
        else
        {
            global = { output.Batch().v, 1, 1 };
        }

        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData NormalizeKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::NORMALIZE);

        const normalize_params& orgParams = static_cast<const normalize_params&>(params);

        DispatchData runInfo;

        runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<normalize_params>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SCALE_TABLE, 0 });

        kd.estimatedTime = estimated_time;

        return{ kd };
    }
}