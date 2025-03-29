// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_iyxo.h"
#include <vector>
#include <utility>
#include <algorithm>

namespace kernel_selector {
// Sub-group size used by "convolution_kernel_bfyx_iyxo" kernel.
constexpr size_t sub_group_size = 16;

ParamsKey ConvolutionKernel_bfyx_iyxo::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_iyxo::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_reqd_subgroup_size();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_iyxo::SetDefault(const convolution_params& cp, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(cp);

    dispatchData.gws[0] = CeilDiv(cp.outputs[0].X().v, sub_group_size) / 4;
    dispatchData.gws[1] = cp.outputs[0].Y().v;
    dispatchData.gws[2] = sub_group_size;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = sub_group_size;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_iyxo::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

bool ConvolutionKernel_bfyx_iyxo::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p) || !ConvolutionCheckInput(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);
    if (params.inputs[0].X().v % 64)
        return false;

    bool bFilterSize = (params.filterSize.x == 5 && params.filterSize.y == 5) ||
                       (params.filterSize.x == 3 && params.filterSize.y == 3 && (params.inputs[0].Feature().v % 4) == 0) ||
                       (params.filterSize.x == 1 && params.filterSize.y == 1);

    bool bStride = (params.stride.x == 1 && params.stride.y == 1);

    if (!bFilterSize || !bStride || (params.outputs[0].Feature().v % 4) != 0 || (params.outputs[0].Batch().v != 1)) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_bfyx_iyxo::GetJitConstants(const convolution_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[2]));

    return jit;
}

KernelsData ConvolutionKernel_bfyx_iyxo::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}

}  // namespace kernel_selector
