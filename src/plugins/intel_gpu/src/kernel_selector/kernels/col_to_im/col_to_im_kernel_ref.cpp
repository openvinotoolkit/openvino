// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col_to_im_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ColToImKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData ColToImKernelRef::SetDefault(const col_to_im_params& params) const {
    CommonDispatchData dispatchData;

    dispatchData.gws = {1, 1, params.outputs[0].Batch().v};
    dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

KernelsData ColToImKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ColToImKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

JitConstants ColToImKernelRef::GetJitConstants(const col_to_im_params& params) const {
    auto jit = Parent::GetJitConstants(params);
    auto input = params.inputs[0];
    auto input_dt = input.GetDType();

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (input.Dimentions() == 5) {
            idx_order = {"batch", "feature", "z", "y", "x"};
        } else if (input.Dimentions() == 4) {
            idx_order = {"batch", "feature", "y", "x"};
        }
        FusedOpsConfiguration conf = {"", idx_order, "in_val", input_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

}  // namespace kernel_selector
