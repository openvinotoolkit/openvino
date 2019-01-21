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


#include "border_kernel_base.h"

#include "kernel_selector_utils.h"


namespace kernel_selector 
{
    JitConstants BorderKernelBase::GetJitConstants(const border_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        jit.AddConstants({
            MakeJitConstant("LT_SIZES",              params.lt_sizes),
            MakeJitConstant("RB_SIZES",              params.rb_sizes),
            MakeJitConstant("BORDER_VALUE",          params.border_value),
            MakeJitConstant(toString(params.b_type), "")
        });

        return jit;
    }

    BorderKernelBase::DispatchData BorderKernelBase::SetDefault(const border_params& params) const
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        std::vector<size_t> global{output.X().v, output.Y().v, output.Batch().v * output.Feature().v};
        const auto& local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData BorderKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::BORDER);

        const auto& prim_params = static_cast<const border_params&>(params); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)

        auto run_info     = SetDefault(prim_params);
        KernelData k_data = KernelData::Default<border_params>(params);

        auto cldnn_jit   = GetJitConstants(prim_params);
        auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
        auto jit         = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = k_data.kernels[0];
        FillCLKernelData(kernel, run_info, params.engineInfo, kernelName, jit, entry_point);

        k_data.estimatedTime = estimated_time;

        return {k_data};
    }
}
