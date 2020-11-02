// Copyright (c) 2016-2020 Intel Corporation
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
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_1x1::SetDefault(const convolution_params& params, int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

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

    dispatchData.efficiency = FORCE_PRIORITY_2;

    return dispatchData;
}

bool ConvolutionKernel_bfyx_1x1::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

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

    if (params.output.Feature().v % 16)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));

    return jit;
}

KernelsData ConvolutionKernel_bfyx_1x1::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
