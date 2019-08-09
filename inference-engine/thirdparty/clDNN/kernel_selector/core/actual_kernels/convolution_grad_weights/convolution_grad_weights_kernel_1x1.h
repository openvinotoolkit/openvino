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

#pragma once

#include "convolution_grad_weights_kernel_base.h"

namespace kernel_selector {

class ConvolutionGradWeightsKernel1x1 : public ConvolutionGradWeightsKernelBase {
public:
    ConvolutionGradWeightsKernel1x1() : ConvolutionGradWeightsKernelBase("convolution_grad_weights_1x1") {}
    virtual ~ConvolutionGradWeightsKernel1x1() {}

    DispatchData SetDefault(const convolution_grad_weights_params& params) const override;
    bool Validate(const Params& p, const optional_params& o) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector