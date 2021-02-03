// Copyright (c) 2016-2021 Intel Corporation
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

#include "kernel_selector_params.h"
#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: decide the folllowing values w.r.t input shape? 
// or fix it as 8 in this kernel?
#define TILE_SIZE_W 8
#define TILE_SIZE_H 8
#define VECTORWIDTH 8
struct permute_params : public base_params {
    permute_params() : base_params(KernelType::PERMUTE) {}

    std::vector<uint16_t> order;
    uint32_t tile_w = TILE_SIZE_W;
    uint32_t tile_h = TILE_SIZE_H;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// permute_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct permute_optional_params : optional_params {
    permute_optional_params() : optional_params(KernelType::PERMUTE) {}
};
}
