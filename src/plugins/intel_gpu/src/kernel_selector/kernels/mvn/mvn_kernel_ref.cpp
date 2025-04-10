// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnableDynamicShapesSupport();
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
        return kernelName + "_across_channels";
    else
        return kernelName + "_within_channels";
}

KernelsData MVNKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority MVNKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
