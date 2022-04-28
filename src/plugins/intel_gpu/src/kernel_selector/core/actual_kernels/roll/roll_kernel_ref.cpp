// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roll_kernel_ref.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const roll_params& params) {
    CommonDispatchData dispatch_data;
    const auto in_layout = params.inputs[0].GetLayout();
    const auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& output = params.outputs[0];

    switch (out_layout) {
    case DataLayout::bfyx:
        dispatch_data.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfzyx:
        dispatch_data.gws = {output.X().v * output.Y().v, output.Z().v, output.Feature().v * output.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case DataLayout::bfwzyx:
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
        GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatch_data;
}

}  // namespace

KernelsData RollKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    auto kernel_data = KernelData::Default<roll_params>(params);
    const auto& new_params = dynamic_cast<const roll_params&>(*kernel_data.params);

    const auto dispatch_data = SetDefault(new_params);
    const auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    const auto roll_specific_jit = GetJitConstants(new_params);
    const auto jit = CreateJit(kernelName, roll_specific_jit, entry_point);

    auto& kernel = kernel_data.kernels[0];

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point);

    return {kernel_data};
}

ParamsKey RollKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableBatching();
    return k;
}

JitConstants RollKernelRef::GetJitConstants(const roll_params& params) const {
    auto jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("SHIFT", params.shift));
    return jit;
}

bool RollKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::ROLL || o.GetType() != KernelType::ROLL) {
        return false;
    }

    const auto& params = dynamic_cast<const roll_params&>(p);
    if (params.inputs.size() != 1) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
