/*
// Copyright (c) 2018 Intel Corporation
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

#include "convolution_grad_weights_kernel_3x3.h"
#include <algorithm>

namespace kernel_selector {

ParamsKey ConvolutionGradWeightsKernel3x3::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableMomentum();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableGradient();
    k.DisableTuning();
    return k;
}

bool ConvolutionGradWeightsKernel3x3::Validate(const Params& p, const optional_params&) const {
    const auto& params = static_cast<const convolution_grad_weights_params&>(p);

    if (params.stride.x != 1 || params.stride.y != 1)
        return false;
    if (params.filterSize.x != 3 || params.filterSize.y != 3)
        return false;
    return true;
}

ConvolutionGradWeightsKernelBase::DispatchData ConvolutionGradWeightsKernel3x3::SetDefault(
    const convolution_grad_weights_params& params) const {
    auto input_features = params.weights.IFM().v;
    auto output_features = params.weights.OFM().v;

    DispatchData kd;

    kd.gws0 = Align(output_features, 16);
    kd.gws1 = input_features;
    kd.gws2 = 1;
    kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
    while (kd.gws0 % kd.lws0 != 0) {
        kd.lws0 -= 16;
    }
    kd.lws1 = 1;
    kd.lws2 = 1;
    kd.effiency = FORCE_PRIORITY_8;
    return kd;
}
}  // namespace kernel_selector
