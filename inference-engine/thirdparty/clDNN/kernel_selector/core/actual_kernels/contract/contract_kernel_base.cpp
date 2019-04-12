// Copyright (c) 2019 Intel Corporation
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


#include "contract_kernel_base.h"

#include "kernel_selector_utils.h"


namespace kernel_selector
{
    JitConstants ContractKernelBase::GetJitConstants(const contract_params& params)
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        const size_t no_dim_flag = 6;
        std::vector<size_t> output_dims(4, no_dim_flag);
        int out_dim = 2;
        for (int i = 3; i >= 0; --i)
        {
            if (std::find(params.reduction_axes.begin(), params.reduction_axes.end(), i) == params.reduction_axes.end())
                output_dims.at(i) = out_dim--;
        }

        if (output_dims[3] != no_dim_flag)
            jit.AddConstants({
                MakeJitConstant("DIM_X", output_dims.at(3))
            });
        if (output_dims[2] != no_dim_flag)
            jit.AddConstants({
                MakeJitConstant("DIM_Y", output_dims.at(2))
            });
        if (output_dims[1] != no_dim_flag)
            jit.AddConstants({
                MakeJitConstant("DIM_F", output_dims.at(1))
            });
        if (output_dims[0] != no_dim_flag)
            jit.AddConstants({
                MakeJitConstant("DIM_B", output_dims.at(0))
            });

        jit.AddConstants({
            MakeJitConstant("REDUCE_X", output_dims.at(3) == no_dim_flag),
            MakeJitConstant("REDUCE_Y", output_dims.at(2) == no_dim_flag),
            MakeJitConstant("REDUCE_F", output_dims.at(1) == no_dim_flag),
            MakeJitConstant("REDUCE_B", output_dims.at(0) == no_dim_flag)
        });

        switch (params.mode)
        {
        case ContractMode::SUM:
            jit.AddConstants({
                MakeJitConstant("REDUCE_SEED", "0"),
                MakeJitConstant("REDUCE_OPERATION(a, b)", "a + b")
            });
            break;
        case ContractMode::PRODUCT:
            jit.AddConstants({
                MakeJitConstant("REDUCE_SEED", "1"),
                MakeJitConstant("REDUCE_OPERATION(a, b)", "a * b")
            });
            break;
        case ContractMode::ALL:
            jit.AddConstants({
                MakeJitConstant("REDUCE_SEED", "1"),
                MakeJitConstant("REDUCE_OPERATION(a, b)", "a && b")
            });
            break;
        case ContractMode::ANY:
            jit.AddConstants({
                MakeJitConstant("REDUCE_SEED", "0"),
                MakeJitConstant("REDUCE_OPERATION(a, b)", "a || b")
            });
            break;
        case ContractMode::MAX:
            jit.AddConstants({
                MakeJitConstant("REDUCE_SEED", "UNIT_VAL_MIN"),
                MakeJitConstant("REDUCE_OPERATION(a, b)", "UNIT_MAX_FUNC(a,b)")
            });
            break;
        }

        return jit;
    }

    ContractKernelBase::DispatchData ContractKernelBase::SetDefault(const contract_params& params)
    {
        const auto& output = params.output;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        std::vector<size_t> global{ output.Feature().v, output.Y().v, output.X().v };
        const auto& local = GetOptimalLocalWorkGroupSizes(global);

        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }

    KernelsData ContractKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options, float estimated_time) const
    {
        assert(params.GetType() == KernelType::CONTRACT);

        const auto& prim_params = static_cast<const contract_params&>(params); // NOLINT(cppcoreguidelines-pro-type-static-cast-downcast)

        auto run_info = SetDefault(prim_params);
        KernelData k_data = KernelData::Default<contract_params>(params);

        auto cldnn_jit = GetJitConstants(prim_params);
        auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = k_data.kernels[0];
        FillCLKernelData(kernel, run_info, params.engineInfo, kernelName, jit, entry_point);
        k_data.estimatedTime = estimated_time;

        return{ k_data };
    }
}
