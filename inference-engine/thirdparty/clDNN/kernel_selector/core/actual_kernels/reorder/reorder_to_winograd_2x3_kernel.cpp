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

#include "reorder_to_winograd_2x3_kernel.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    ParamsKey ReorderToWinograd2x3Kernel::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::winograd_2x3_s1_data);
        k.EnableWinogradReorder();
        k.EnableDifferentTypes();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    JitConstants ReorderToWinograd2x3Kernel::GetJitConstants(const reorder_params& params) const
    {
        auto jit = ReorderKernelBase::GetJitConstants(params);

        jit.AddConstant(MakeJitConstant("INPUT0_OFFSET_SIZE_X", params.winograd_input_offset_x));
        jit.AddConstant(MakeJitConstant("INPUT0_OFFSET_SIZE_Y", params.winograd_input_offset_y));

        return jit;
    }

    ReorderToWinograd2x3Kernel::DispatchData ReorderToWinograd2x3Kernel::SetDefault(const reorder_params& params) const
    {
        DispatchData kd;

        const auto& input = params.inputs[0];
        const auto& output = params.output;

        kd.gws0 = static_cast<size_t>(input.Feature().v * input.Batch().v);
        kd.gws1 = static_cast<size_t>(params.winograd_nr_tiles_x);
        kd.gws2 = static_cast<size_t>(output.Y().v);

        kd.lws0 = input.Feature().v > 32 ? 32 : static_cast<size_t>(input.Feature().v);
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    KernelsData ReorderToWinograd2x3Kernel::GetKernelsData(const Params& params, const optional_params& options) const
    {
        const reorder_params& orgParams = static_cast<const reorder_params&>(params);
        return GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_6);
    }
}