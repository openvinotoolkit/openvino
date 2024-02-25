// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ctc_loss_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const ctc_loss_params& kernel_params) {
    CommonDispatchData dispatch_data;
    const auto& output = kernel_params.outputs.front();

    dispatch_data.gws = {output.Batch().v, 1, 1};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo);

    return dispatch_data;
}

}  // namespace

KernelsData CTCLossKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<ctc_loss_params>(params);
    const auto& kernel_params = dynamic_cast<const ctc_loss_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
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
                     static_cast<int>(kernel_params.inputs.size()));

    return {kernel_data};
}

ParamsKey CTCLossKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::INT32);
    key.EnableInputDataType(Datatype::INT64);
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableDifferentTypes();
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

bool CTCLossKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::CTC_LOSS) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const ctc_loss_params&>(params);
    if (kernel_params.inputs.size() != 4 && kernel_params.inputs.size() != 5) {
        return false;
    }

    return true;
}

JitConstants CTCLossKernelRef::GetJitConstants(const ctc_loss_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("PREPROCESS_COLLAPSE_REPEATED", kernel_params.preprocess_collapse_repeated),
        MakeJitConstant("CTC_MERGE_REPEATED", kernel_params.ctc_merge_repeated),
        MakeJitConstant("UNIQUE", kernel_params.unique),
    });

    return jit_constants;
}

}  // namespace kernel_selector
