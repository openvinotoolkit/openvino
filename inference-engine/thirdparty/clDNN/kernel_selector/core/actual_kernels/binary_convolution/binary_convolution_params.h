/*
// Copyright (c) 2019 Intel Corporation
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

#include "weight_bias_params.h"
#include <vector>
#include <string>

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// binary_convolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct binary_convolution_params : public weight_bias_params {
    binary_convolution_params() : weight_bias_params(KernelType::BINARY_CONVOLUTION) {}

    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    Datatype out_dt = Datatype::UNSUPPORTED;
    uint32_t split = 1;
    bool depthwise_separable_opt = false;
    float pad_value = 0.0f;
    uint32_t groups = 1;

    std::string to_string() const override;
    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct binary_convolution_optional_params : weight_bias_optional_params {
    binary_convolution_optional_params() : weight_bias_optional_params(KernelType::BINARY_CONVOLUTION) {}
};

}  // namespace kernel_selector
