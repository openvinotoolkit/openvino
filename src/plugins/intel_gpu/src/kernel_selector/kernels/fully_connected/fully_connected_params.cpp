// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_params.h"
#include "kernel_selector_utils.h"

using namespace cldnn;

namespace kernel_selector {

size_t fully_connected_params::hash() const {
    auto seed = weight_bias_params::hash();
    seed = hash_combine(seed, quantization);
    return seed;
}

}  // namespace kernel_selector