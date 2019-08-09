/*
// Copyright (c) 2016 Intel Corporation
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

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fully_connected_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct fully_connected_params : public weight_bias_params {
    fully_connected_params() : weight_bias_params(KernelType::FULLY_CONNECTED) {}

    bool int8_quantization = false;
    bool output_calibration = false;
    float input_quantization_factor = 1.0f;
    float output_quantization_factor = 1.0f;

    MultiDataTensor weights_quantization_factors;
    MultiDataTensor output_calibration_factors;

    virtual ParamsKey GetParamsKey() const {
        ParamsKey k = weight_bias_params::GetParamsKey();

        if (int8_quantization) {
            k.EnableInt8Quantization();
        }

        if (output_calibration) {
            k.EnableOutputCalibration();
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