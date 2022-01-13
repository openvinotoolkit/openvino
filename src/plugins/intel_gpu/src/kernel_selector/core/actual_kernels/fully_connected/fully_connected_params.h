// Copyright (C) 2018-2021 Intel Corporation
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

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        k.EnableQuantization(quantization);

        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fully_connected_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fully_connected_optional_params : weight_bias_optional_params {
    fully_connected_optional_params() : weight_bias_optional_params(KernelType::FULLY_CONNECTED) {}
};
}  // namespace kernel_selector
