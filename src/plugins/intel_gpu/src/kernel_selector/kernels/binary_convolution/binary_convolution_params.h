// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    std::string to_cache_string_v2() const override;
    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct binary_convolution_optional_params : weight_bias_optional_params {
    binary_convolution_optional_params() : weight_bias_optional_params(KernelType::BINARY_CONVOLUTION) {}
};

}  // namespace kernel_selector
