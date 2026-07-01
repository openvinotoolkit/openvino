// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault() {
    CommonDispatchData dispatch_data;
    dispatch_data.gws = {1, 1, 1};
    dispatch_data.lws = {1, 1, 1};
    return dispatch_data;
}

}  // namespace

KernelsData HistcKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<histc_params>(params);
    const auto& kernel_params = static_cast<const histc_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault();
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, {}, false, false, 1);

    return {kernel_data};
}

ParamsKey HistcKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableBatching();
    return key;
}

bool HistcKernelRef::Validate(const Params& params) const {
    if (!KernelBaseOpenCL::Validate(params)) {
        return false;
    }

    if (params.GetType() != KernelType::HISTC) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& kernel_params = static_cast<const histc_params&>(params);
    if (kernel_params.inputs.size() != 1 || kernel_params.bins < 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return Tensor::SimpleLayout(kernel_params.inputs[0].GetLayout());
}

JitConstants HistcKernelRef::GetJitConstants(const histc_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);
    jit_constants.AddConstants({MakeJitConstant("BINS", kernel_params.bins),
                                MakeJitConstant("MIN_VAL", static_cast<float>(kernel_params.min_val)),
                                MakeJitConstant("MAX_VAL", static_cast<float>(kernel_params.max_val))});

    if (kernel_params.has_dynamic_tensors()) {
        const auto& input = kernel_params.inputs[0];
        DimensionAccessHelperJit dims(input);
        const std::string total_elements = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.w(), dims.f(), dims.b()});
        jit_constants.AddConstant(MakeJitConstant("TOTAL_ELEMENTS", total_elements));
    } else {
        jit_constants.AddConstant(MakeJitConstant("TOTAL_ELEMENTS", kernel_params.inputs[0].LogicalSize()));
    }

    return jit_constants;
}

}  // namespace kernel_selector
