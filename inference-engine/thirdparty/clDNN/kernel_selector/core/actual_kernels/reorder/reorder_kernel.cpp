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

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"
 
namespace kernel_selector 
{
    ParamsKey ReorderKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::UINT8);
        k.EnableInputDataType(Datatype::INT8);
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::UINT8);
        k.EnableDifferentTypes();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    JitConstants ReorderKernelRef::GetJitConstants(const reorder_params& params) const
    {
        auto jit = ReorderKernelBase::GetJitConstants(params);
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
        return jit;
    }

    KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        const reorder_params& orgParams = static_cast<const reorder_params&>(params);
        return GetCommonKernelsData(orgParams, options, DONT_USE_IF_HAVE_SOMETHING_ELSE);
    }
}