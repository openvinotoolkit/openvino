// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deformable_convolution_kernel_bfyx_conv.h"

namespace kernel_selector {

ParamsKey DeformableConvolutionKernel_bfyx_conv::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.DisableTuning();
    k.EnableGroupedConvolution();
    k.EnableDeformableMode();
    k.EnableDeformableMask();
    k.EnableBilinearInterpolationPad();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

DeformableConvolutionKernel_bfyx_conv::DispatchData DeformableConvolutionKernel_bfyx_conv::SetDefault(const convolution_params& params,
                                                                                                      int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = CeilDiv(x * y, 16);
    dispatchData.gws[1] = Align(f, 16);
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 16;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority DeformableConvolutionKernel_bfyx_conv::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_2;
}

JitConstants DeformableConvolutionKernel_bfyx_conv::GetJitConstants(const convolution_params& params,
                                                                    const DispatchData& /*dispatchData*/) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", 16));
    jit.AddConstant(MakeJitConstant("INPUT_CHANNELS", params.inputs[0].Feature().v / params.weights.X().v / params.weights.Y().v));
    return jit;
}

KernelsData DeformableConvolutionKernel_bfyx_conv::GetKernelsData(const Params& params,
                                                                 const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
