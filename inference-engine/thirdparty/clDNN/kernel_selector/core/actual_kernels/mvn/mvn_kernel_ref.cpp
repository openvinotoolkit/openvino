// Copyright (c) 2018-2020 Intel Corporation
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


#include "mvn_kernel_ref.h"

#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey MVNKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNNormalizeVariance();
    return k;
}

JitConstants MVNKernelRef::GetJitConstants(const mvn_params& params, DispatchData dispatchData) const {
    auto jits = Parent::GetJitConstants(params, dispatchData);

    auto activation_dt = GetActivationType(params);
    jits.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = { "b", "f", "y", "x" };
        } else if (params.inputs[0].GetDims().size() == 5) {
            idx_order = { "b", "f", "z", "y", "x" };
        }
        auto conf = FusedOpsConfiguration("", idx_order, "result", activation_dt);
        jits.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }
    return jits;
}

std::string MVNKernelRef::GetKernelName(const mvn_params& params) const {
    if (params.mvnMode == MVNMode::ACROSS_CHANNELS)
        return kernelName + "_accross_channels";
    else
        return kernelName + "_within_channels";
}

KernelsData MVNKernelRef::GetKernelsData(const Params& params, const optional_params& optParams) const {
    return GetCommonKernelsData(params, optParams, FORCE_PRIORITY_9);
}
}  // namespace kernel_selector
