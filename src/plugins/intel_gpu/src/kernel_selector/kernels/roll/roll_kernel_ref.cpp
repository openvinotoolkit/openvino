// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roll_kernel_ref.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const roll_params& kernel_params) {
    CommonDispatchData dispatch_data;
    const auto in_layout = kernel_params.inputs.front().GetLayout();
    const auto& output = kernel_params.outputs.front();
    const auto out_layout = output.GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    switch (output.Dimentions()) {
    case 4:
        dispatch_data.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case 5:
        dispatch_data.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case 6:
        dispatch_data.gws = {output.X().v * output.Y().v,
                             output.Z().v * output.W().v,
                             output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    default:
        throw std::invalid_argument("Unsupported data layout for roll primitive");
    }

    dispatch_data.lws =
        GetOptimalLocalWorkGroupSizes(dispatch_data.gws, kernel_params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatch_data;
}

}  // namespace

KernelsData RollKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<roll_params>(params);
    const auto& kernel_params = dynamic_cast<const roll_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point);

    return {kernel_data};
}

ParamsKey RollKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableAllInputDataType();
    key.EnableAllOutputDataType();
    key.EnableAllInputLayout();
    key.EnableAllOutputLayout();
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

JitConstants RollKernelRef::GetJitConstants(const roll_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);
    jit_constants.AddConstant(MakeJitConstant("SHIFT", kernel_params.shift));
    return jit_constants;
}

bool RollKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::ROLL) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& kernel_params = dynamic_cast<const roll_params&>(params);
    if (kernel_params.inputs.size() != 1) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

}  // namespace kernel_selector
