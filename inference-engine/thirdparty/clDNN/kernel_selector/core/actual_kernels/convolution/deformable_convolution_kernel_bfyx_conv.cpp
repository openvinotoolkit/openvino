/*
// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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
    k.EnableLocalConvolution();
    k.EnableGroupedConvolution();
    k.EnableDeformableMode();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

DeformableConvolutionKernel_bfyx_conv::DispatchData DeformableConvolutionKernel_bfyx_conv::SetDefault(const convolution_params& params,
                                                                                                      int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = CeilDiv(x * y, 16);
    kd.gws1 = Align(f, 16);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = 16;
    kd.lws2 = 1;

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}

JitConstants DeformableConvolutionKernel_bfyx_conv::GetJitConstants(const convolution_params& params,
                                                                    const DispatchData& /*kd*/) const {
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
