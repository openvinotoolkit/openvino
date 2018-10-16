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

#include "activation_kernel_opt.h"
#include "kernel_selector_utils.h" 

namespace kernel_selector {

    ParamsKey ActivationKernelOpt::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableBatching();
        k.EnableGradient();
        return k;
    }

    ActivationKernelOpt::Parent::DispatchData ActivationKernelOpt::SetDefault(const activation_params& params) const
    {
        auto runInfo = Parent::SetDefault(params);

        const auto totalSize = params.inputs[0].LogicalSize();

        std::vector<size_t> global = { totalSize/NUM_COLS_WI };
        std::vector<size_t> local = GetOptimalLocalWorkGroupSizes(global);

        runInfo.gws0 = global[0];
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = local[0];
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        runInfo.effiency = FORCE_PRIORITY_6;

        return runInfo;
    }

    bool ActivationKernelOpt::Validate(const Params& p, const optional_params& o) const
    {
        if (p.GetType() != KernelType::ACTIVATION ||
            o.GetType() != KernelType::ACTIVATION)
        {
            return false;
        }

        const activation_params& params = static_cast<const activation_params&>(p);

        const auto totalSize = params.inputs[0].LogicalSize();
        if ((totalSize % NUM_COLS_WI) != 0 ||
            (params.inputs[0].GetFirstElementOffset() % NUM_COLS_WI) != 0 ||
            (params.output.GetFirstElementOffset() % NUM_COLS_WI) != 0)
        {
            return false;
        }

        if (params.gradient)
        {
            if (params.inputs[0].GetLayout() != params.inputs[1].GetLayout())
                return false;
        }

        return true;
    }

    JitConstants ActivationKernelOpt::GetJitConstants(const activation_params& params, DispatchData kd) const
    {
        auto jit = ActivationKernelBase::GetJitConstants(params, kd);

        jit.AddConstant(MakeJitConstant("NUM_COLS_WI", NUM_COLS_WI));

        return jit;
    }

    KernelsData ActivationKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }
}