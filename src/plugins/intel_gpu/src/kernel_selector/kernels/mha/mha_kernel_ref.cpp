// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey MHAKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData MHAKernelRef::SetDefault(const mha_params& params) const {
    CommonDispatchData dispatchData;

    /* Note: even for ref implementation, we can parallelize f-axis */
    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = dispatchData.gws;

    return dispatchData;
}

JitConstants MHAKernelRef::GetJitConstants(const mha_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

bool MHAKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::MHA || o.GetType() != KernelType::MHA) {
        return false;
    }

    /* FIXME: fill here to allow SD-2.1 only */

    return true;
}

KernelsData MHAKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<mha_params>(params);
    mha_params& newParams = *static_cast<mha_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 3, GetFusedPrimitiveInputsCount(params));

    return { kd };}

}  // namespace kernel_selector
