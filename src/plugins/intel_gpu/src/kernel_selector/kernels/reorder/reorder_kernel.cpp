// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ReorderKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableSurfaceInputSupport();
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants ReorderKernelRef::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    if (params.truncate) {
        jit.AddConstant(MakeJitConstant("CONVERT_TRUNCATE", true));
    }
    jit.Merge(GetTensorFriendlyWorkGroupsJit(params.inputs[0]));

    if (params.surface_input)
        jit.AddConstant(MakeJitConstant("SURFACE_INPUT", true));

    return jit;
}

KernelsData ReorderKernelRef::GetKernelsData(const Params& params) const {
    const reorder_params& orgParams = static_cast<const reorder_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
