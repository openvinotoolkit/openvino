// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_params.h"
#include <string>
#include <vector>

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct convolution_params : public weight_bias_zero_point_params {
    using parent = weight_bias_zero_point_params;

    convolution_params() : parent(KernelType::CONVOLUTION) {}
    uSize filterSize;
    uSize stride;
    uSize dilation;
    uSize padding;
    uint32_t split = 1;
    bool depthwise_separable_opt = false;
    bool transposed = false;
    QuantizationType quantization = QuantizationType::NONE;
    bool deformable_mode = false;
    uint32_t groups = 1;
    uSize kernelSize;
    uint32_t deformable_groups = 1;
    bool bilinear_interpolation_pad {false};
    bool deformable_mask_enabled {false};

    std::string to_string() const override;
    std::string to_cache_string_v2() const override;
    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// convolution_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct convolution_optional_params : weight_bias_optional_params {
    convolution_optional_params() : weight_bias_optional_params(KernelType::CONVOLUTION) {}
};

}  // namespace kernel_selector
