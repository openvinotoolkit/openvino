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

#include "weight_bias_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// training_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct training_params : public weight_bias_params {
    explicit training_params(KernelType kt) : weight_bias_params(kt) {}

    bool use_momentum = false;
    float weights_decay;
    float momentum_factor;

    ParamsKey GetParamsKey() const override;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// training_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct training_optional_params : weight_bias_optional_params {
protected:
    explicit training_optional_params(KernelType kt) : weight_bias_optional_params(kt) {}
};

}  // namespace kernel_selector