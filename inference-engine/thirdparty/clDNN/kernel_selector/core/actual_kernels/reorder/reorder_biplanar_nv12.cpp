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


#include "reorder_biplanar_nv12.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey reorder_biplanar_nv12::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::nv12);
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants reorder_biplanar_nv12::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));
    return jit;
}

KernelsData reorder_biplanar_nv12::GetKernelsData(const Params& params, const optional_params& options) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    if (orgParams.inputs.size() != 2) {
        return {};
    }
    KernelsData kd = GetCommonKernelsData(orgParams, options, FORCE_PRIORITY_9);
    kd[0].kernels[0].arguments = GetArgsDesc(2, false, false);
    return kd;
}
}  // namespace kernel_selector
