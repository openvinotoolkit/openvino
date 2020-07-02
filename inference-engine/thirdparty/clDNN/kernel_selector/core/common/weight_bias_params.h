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

#include "kernel_selector_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// weight_bias_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct weight_bias_params : public base_params {
    explicit weight_bias_params(KernelType kt) : base_params(kt) {}

    WeightsTensor weights;
    MultiDataTensor bias;

    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// weight_bias_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct weight_bias_zero_point_params : public weight_bias_params {
    using weight_bias_params::weight_bias_params;

    MultiDataTensor weights_zero_points;
    MultiDataTensor activations_zero_points;
    MultiDataTensor compensation;

    bool HasCompensation() const { return !compensation.empty(); }
    std::string to_cache_string_v2() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// weight_bias_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct weight_bias_optional_params : optional_params {
protected:
    explicit weight_bias_optional_params(KernelType kt) : optional_params(kt) {}
};

}  // namespace kernel_selector
