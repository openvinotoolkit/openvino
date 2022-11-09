// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eye_kernel_ref.h"

#include <kernel_selector_utils.h>

#include <vector>

namespace kernel_selector {

KernelsData EyeKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<eye_params>(params);
    const eye_params& new_params = dynamic_cast<eye_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    auto slice_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, slice_specific_jit, entry_point);

    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo, kernelName, jit, entry_point);
    return {kernel_data};
}

KernelsPriority EyeKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey EyeKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    // Inputs for Eye operation can be only integer types but in case of blocked layout and when output is float,
    // the inputs are transformed to f32.
    // That is the reason why the operation below is present
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool EyeKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::EYE || o.GetType() != KernelType::EYE) {
        return false;
    }

    const eye_params& params = dynamic_cast<const eye_params&>(p);
    if (params.inputs.empty())
        return false;

    return true;
}

JitConstants EyeKernelRef::GetJitConstants(const eye_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("DIAGONAL", params.diagonal_index));
    return jit;
}

CommonDispatchData EyeKernelRef::SetDefault(const eye_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;
    dispatchData.gws = {params.outputs[0].Batch().v,
                        params.outputs[0].Feature().v,
                        params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v};

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

}  // namespace kernel_selector
