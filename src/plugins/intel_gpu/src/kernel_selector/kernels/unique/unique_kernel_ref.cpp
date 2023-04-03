// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unique_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

KernelsData UniqueKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    auto kernel_data = KernelData::Default<unique_params>(params);
    const auto& kernel_params = dynamic_cast<const unique_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, options);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     kernel_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     kernel_params.outputs.size());

    return {kernel_data};
}

ParamsKey UniqueKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableDifferentTypes();
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

bool UniqueKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (params.GetType() != KernelType::UNIQUE || options.GetType() != KernelType::UNIQUE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const unique_params&>(params);
    if (kernel_params.inputs.size() != 1) {
        return false;
    }
    if (kernel_params.outputs.size() != 4 && kernel_params.outputs.size() != 5) {
        return false;
    }

    return true;
}

JitConstants UniqueKernelRef::GetJitConstants(const unique_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("FLATTENED", kernel_params.flattened),
        MakeJitConstant("AXIS", kernel_params.axis),
        MakeJitConstant("SORTED", kernel_params.sorted),
    });

    return jit_constants;
}

CommonDispatchData UniqueKernelRef::SetDefault(const unique_params& kernel_params) {
    CommonDispatchData dispatch_data;

    if (kernel_params.flattened) {
        // For now we run flattened case only in one thread
        // TODO: Parallelize flattened case
        dispatch_data.gws = {1, 1, 1};
        dispatch_data.lws = {1, 1, 1};
    }

    return dispatch_data;
}

}  // namespace kernel_selector
