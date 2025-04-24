// Copyright (C) 2018-2025 Intel Corporation
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
    uSize padding_begin;
    uSize padding_end;
    bool transposed = false;
    QuantizationType quantization = QuantizationType::NONE;
    bool deformable_mode = false;
    uint32_t groups = 1;
    uint32_t deformable_groups = 1;
    bool bilinear_interpolation_pad {false};
    bool deformable_mask_enabled {false};
    bool has_explicit_paddings {false};
    DataTensor intermediate_tensor;

    std::string to_string() const override;
    std::string to_cache_string_v2() const override;
    ParamsKey GetParamsKey() const override;
};

}  // namespace kernel_selector
