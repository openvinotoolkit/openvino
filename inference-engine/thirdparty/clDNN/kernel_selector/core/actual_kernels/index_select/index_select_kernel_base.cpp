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


#include "index_select_kernel_base.h"

#include "kernel_selector_utils.h"


namespace kernel_selector 
{
    JitConstants IndexSelectKernelBase::GetJitConstants(const index_select_params& params)
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        jit.AddConstant(MakeJitConstant(toString(params.axis), ""));

        return jit;
    }

    IndexSelectKernelBase::DispatchData IndexSelectKernelBase::SetDefault(const index_select_params& params)
    {
        const auto& output = params.output;
        const auto& indices = params.inputs.at(1);
        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        std::vector<size_t> global;
        if (params.axis == IndexSelectAxis::BATCH)
        {
            global = { 1, indices.X().v, output.Feature().v };
        }
        else if (params.axis == IndexSelectAxis::X || params.axis == IndexSelectAxis::Y)
        {
            global = { output.Batch().v, indices.X().v, output.Feature().v };
        }
        else if(params.axis == IndexSelectAxis::FEATURE)
        {
            global = { output.Batch().v, indices.X().v, output.Y().v };
        }
        const auto& local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData IndexSelectKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::INDEX_SELECT);

        const auto& prim_params = static_cast<const index_select_params&>(params); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)
        
        auto run_info     = SetDefault(prim_params);
        KernelData k_data = KernelData::Default<index_select_params>(params);

        auto cldnn_jit   = GetJitConstants(prim_params);
        auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
        auto jit         = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = k_data.kernels[0];
        FillCLKernelData(kernel, run_info, params.engineInfo, kernelName, jit, entry_point, ROUND_ROBIN, false, false, (uint32_t)prim_params.inputs.size());

        k_data.estimatedTime = estimated_time;

        return {k_data};
    }
}
