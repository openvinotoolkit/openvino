// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "weight_bias_params.h"
#include "openvino/core/partial_shape.hpp"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fully_connected_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fully_connected_params : public weight_bias_params {
    fully_connected_params() : weight_bias_params(KernelType::FULLY_CONNECTED) {}

    QuantizationType quantization = QuantizationType::NONE;
    bool new_shape_infer = false;
    ov::PartialShape input_shape;
    ov::PartialShape weights_shape;
    ov::PartialShape output_shape;

    ParamsKey GetParamsKey() const override {
        ParamsKey k = weight_bias_params::GetParamsKey();

        k.EnableQuantization(quantization);

        if (new_shape_infer) {
            k.EnableNewShapeInfer();
        }

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
