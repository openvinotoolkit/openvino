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

#include "reorder_kernel_byxf_f32_to_byx8_f4_i8.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    ParamsKey reorder_kernel_byxf_f32_to_byx8_f4_i8::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableDifferentTypes();
        k.EnableInputLayout(DataLayout::byxf);
        k.EnableOutputLayout(DataLayout::byx8_f4);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    bool reorder_kernel_byxf_f32_to_byx8_f4_i8::Validate(const Params& p, const optional_params& o) const
    {
        if (!ReorderKernelBase::Validate(p, o))
        {
            return false;
        }

        const reorder_params& params = static_cast<const reorder_params&>(p);

        if (params.output.X().v % 16 != 0)
            return false;

        if (params.inputs[0].Feature().v != 3)
            return false;

        return true;
    }

    reorder_kernel_byxf_f32_to_byx8_f4_i8::DispatchData reorder_kernel_byxf_f32_to_byx8_f4_i8::SetDefault(const reorder_params& params) const
    {
        DispatchData kd;

        const auto& input = params.inputs[0];

        kd.gws0 = input.X().v;
        kd.gws1 = input.Y().v;
        kd.gws2 = input.Batch().v;

        kd.lws0 = 16;
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    JitConstants reorder_kernel_byxf_f32_to_byx8_f4_i8::GetJitConstants(const reorder_params& params) const
    {
        auto jit = ReorderKernelBase::GetJitConstants(params);
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        return jit;
    }

    KernelsData reorder_kernel_byxf_f32_to_byx8_f4_i8::GetKernelsData(const Params& params, const optional_params& options) const
    {
        const reorder_params& orgParams = static_cast<const reorder_params&>(params);
        return GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_5);
    }
}