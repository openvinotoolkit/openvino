// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants ActivationKernelRef::GetJitConstants(const activation_params& params, DispatchData dispatchData) const {
    auto jit = ActivationKernelBase::GetJitConstants(params, dispatchData);
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

KernelsData ActivationKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ActivationKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

bool ActivationKernelRef::Validate(const Params& p) const {
    if (!Parent::Validate(p)) return false;
    const auto& params = static_cast<const activation_params&>(p);
    if (params.inputs[0].GetDims().size() != params.outputs[0].GetDims().size())
        return false;

    return true;
}
}  // namespace kernel_selector
