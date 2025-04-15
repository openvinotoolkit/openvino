// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fully_connected_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fully_connected_params : public weight_bias_params {
    fully_connected_params() : weight_bias_params(KernelType::FULLY_CONNECTED) {}

    QuantizationType quantization = QuantizationType::NONE;
    size_t dynamic_quantization_group_size = 0;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        k.EnableQuantization(quantization);

        return k;
    }
};

}  // namespace kernel_selector
