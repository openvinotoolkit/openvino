// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_direct_10_12_16.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_bfyx_Direct_10_10_12::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_Direct_10_10_12::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_broadcast();

    return k;
}

JitConstants ConvolutionKernel_bfyx_Direct_10_10_12::GetJitConstants(const convolution_params& cp,
                                                                     const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstantsWithLoopUnroll(cp, dispatchData);

    jit.AddConstants({
        MakeJitConstant("ALIGNED_OFM", RoundUp(cp.outputs[0].Feature().v / cp.groups, dispatchData.gemmStyle.subBlockDimN) * cp.groups),
        MakeJitConstant("ALIGNED_OFM_PER_GROUP", RoundUp(cp.outputs[0].Feature().v / cp.groups, dispatchData.gemmStyle.subBlockDimN)),
        MakeJitConstant("DX", dispatchData.gemmStyle.globalWorkSizeDX),
        MakeJitConstant("DY", dispatchData.gemmStyle.globalWorkSizeDY),
        MakeJitConstant("KERNEL_SLICE_DIV2", (cp.filterSize.x * cp.filterSize.y) / 2),
        MakeJitConstant("RIGHT_PARTIAL_TILE_K", cp.outputs[0].X().v % dispatchData.gemmStyle.globalWorkSizeDX),
        MakeJitConstant("INPUT_BUFFER_WIDTH_PADDED", ""),  // TODO: enable non padding path again
        MakeJitConstant("INPUT_BUFFER_HEIGHT_PADDED", ""),
    });

    return jit;
}

ConvolutionKernel_bfyx_Direct_10_10_12::DispatchData ConvolutionKernel_bfyx_Direct_10_10_12::SetDefault(const convolution_params& arg,
                                                                                                        int) const {
    DispatchData dispatchData = Parent::SetDefault(arg);

    constexpr uint32_t TILE_N = 16;

    if (arg.filterSize.x == 5) {
        dispatchData.gemmStyle = {1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 4, 1};
    } else {
        dispatchData.gemmStyle = {1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 3, 1};
    }

    dispatchData.gws[0] = RoundUp(arg.outputs[0].X().v, dispatchData.gemmStyle.globalWorkSizeDX) / dispatchData.gemmStyle.globalWorkSizeDX;
    dispatchData.gws[1] = RoundUp(arg.outputs[0].Y().v, dispatchData.gemmStyle.globalWorkSizeDY) / dispatchData.gemmStyle.globalWorkSizeDY;
    dispatchData.gws[2] = RoundUp(arg.outputs[0].Feature().v / arg.groups, TILE_N) * arg.outputs[0].Batch().v * arg.groups;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = TILE_N;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_Direct_10_10_12::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}

bool ConvolutionKernel_bfyx_Direct_10_10_12::Validate(const Params& p) const {
    if (!Parent::Validate(p) || !ConvolutionCheckInput(p)) {
        return false;
    }

    const convolution_params& cp = static_cast<const convolution_params&>(p);

    const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
    const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
    const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
    const bool bFilterOK = bFilter3x3 || bFilter5x5;

    if (!bFilterOK || !bStrideOK) {
        return false;
    }

    return true;
}

KernelsData ConvolutionKernel_bfyx_Direct_10_10_12::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
