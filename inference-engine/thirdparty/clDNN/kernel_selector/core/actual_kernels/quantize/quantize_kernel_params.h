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


#pragma once

#include "common_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_params : public base_params {
    quantize_params() : base_params(KernelType::QUANTIZE),
    levels(0), packed_binary_output(false) {}

    int levels;
    bool packed_binary_output;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        if (packed_binary_output)
            k.EnableQuantizePackedBinaryOutput();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_optional_params : optional_params {
    quantize_optional_params() : optional_params(KernelType::QUANTIZE) {}
};

}  // namespace kernel_selector
