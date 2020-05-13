// Copyright (c) 2016-2019 Intel Corporation
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


#include "activation_kernel_ref.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey ActivationKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableActivationAdditionalParamsAsInput();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableGradient();
    return k;
}

JitConstants ActivationKernelRef::GetJitConstants(const activation_params& params, DispatchData kd) const {
    auto jit = ActivationKernelBase::GetJitConstants(params, kd);
    auto input_dt = params.inputs[0].GetDType();

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = {"batch", "feature", "y", "x"};
        } else if (params.inputs[0].GetDims().size() == 5) {
            idx_order = {"batch", "feature", "z", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "dst", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, input_dt, "_KERNEL"));
    return jit;
}

KernelsData ActivationKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
