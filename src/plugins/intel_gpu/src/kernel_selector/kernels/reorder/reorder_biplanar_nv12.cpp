// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

KernelsData reorder_biplanar_nv12::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    if (orgParams.inputs.size() != 2) {
        return {};
    }
    KernelsData kd = GetCommonKernelsData(orgParams);
    kd[0].kernels[0].params.arguments = GetArgsDesc(2, false, false);
    return kd;
}

KernelsPriority reorder_biplanar_nv12::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
