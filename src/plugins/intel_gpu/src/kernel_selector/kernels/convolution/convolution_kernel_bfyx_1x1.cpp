// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_1x1.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_bfyx_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_1x1::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(x * y, 16) / 16;
    dispatchData.gws[1] = Align(f, 16);
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 16;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_1x1::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool ConvolutionKernel_bfyx_1x1::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const bool bOutputSizes = output.X().v != input.X().v || output.Y().v != input.Y().v;
    const bool bPad = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 || input.Batch().pad.Total() != 0;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bInputSizes = input.GetLayout() == DataLayout::bfyx && (input.X().v * input.Y().v != 16 || (input.Feature().v % 8) != 0);

    if (bOutputSizes || bPad || bFilterSize || bStride || bInputSizes) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_1x1::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.outputs[0].Feature().v % 16)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));

    return jit;
}

KernelsData ConvolutionKernel_bfyx_1x1::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
