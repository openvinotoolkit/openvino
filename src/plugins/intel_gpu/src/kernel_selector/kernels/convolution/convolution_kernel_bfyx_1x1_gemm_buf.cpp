// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_1x1_gemm_buf.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_bfyx_1x1_gemm_buf::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_1x1_gemm_buf::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1_gemm_buf::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = Align(f, 16);
    dispatchData.gws[1] = CeilDiv(x * y, 16);
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_1x1_gemm_buf::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

bool ConvolutionKernel_bfyx_1x1_gemm_buf::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];

    const bool bPad = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Feature().pad.Total() != 0 || input.Batch().pad.Total() != 0;
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bIFMSize = input.Feature().v % 32 != 0;

    if (bPad || bFilterSize || bStride || bIFMSize) {
        return false;
    }

    if (!params.engineInfo.supports_image)
        return false;

    return true;
}

JitConstants ConvolutionKernel_bfyx_1x1_gemm_buf::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    const auto& out = params.outputs[0];
    const auto& input = params.inputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;

    auto num_whole_groups_y = x * y / (16);
    auto num_whole_subgroups_y = (x * y - num_whole_groups_y * 16) / 16;
    auto last_local_y = x * y - (num_whole_groups_y + num_whole_subgroups_y) * 16;

    jit.AddConstant(MakeJitConstant("TX", 16));
    jit.AddConstant(MakeJitConstant("TY", 1));
    jit.AddConstant(MakeJitConstant("M", x * y));
    jit.AddConstant(MakeJitConstant("K", input.Feature().v));
    jit.AddConstant(MakeJitConstant("N", out.Feature().v));
    jit.AddConstant(MakeJitConstant("TILE_M", 16));
    jit.AddConstant(MakeJitConstant("TILE_N", 16));
    jit.AddConstant(MakeJitConstant("K8", (input.Feature().v >> 3)));
    jit.AddConstant(MakeJitConstant("NUM_WHOLE_GROUPS_Y", num_whole_groups_y));
    jit.AddConstant(MakeJitConstant("NUM_WHOLE_SUBGROUPS_Y", num_whole_subgroups_y));
    jit.AddConstant(MakeJitConstant("LAST_LOCAL_Y", last_local_y));

    return jit;
}

KernelsData ConvolutionKernel_bfyx_1x1_gemm_buf::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
