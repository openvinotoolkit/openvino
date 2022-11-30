// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "weight_bias_params.h"
#include "kernel_selector_common.h"
#include "intel_gpu/runtime/utils.hpp"
#include "kernel_selector_utils.h"
#include <sstream>

namespace kernel_selector {
ParamsKey weight_bias_params::GetParamsKey() const {
    ParamsKey k = base_params::GetParamsKey();

    k.EnableInputWeightsType(weights.GetDType());

    // not needed - can be changed by reorder params
    // k.EnableWeightsLayout(weights.layout);

    assert(bias.size() <= 1);

    if (bias.empty()) {
        k.EnableNonBiasTerm();
    } else if (bias[0].GetLayout() == DataLayout::bf || bias[0].GetLayout() == DataLayout::fb) {
        k.EnableBiasPerFeature();
    } else if (bias[0].GetLayout() == outputs[0].GetLayout()) {
        k.EnableBiasPerOutput();
    }

    return k;
}

std::string weight_bias_zero_point_params::to_cache_string_v2() const {
    std::stringstream s;

    s << weight_bias_params::to_cache_string_v2();
    if (!activations_zero_points.empty())
        s << ";activation_zp";
    if (!weights_zero_points.empty())
        s << ";weights_zp";
    if (HasCompensation())
        s << ";compensation";

    return s.str();
}

size_t weight_bias_params::hash() const {
    size_t seed = base_params::hash();
    seed = hash_combine_wt(seed, weights);
    for (auto& dt : bias)
        seed = hash_combine_dt(seed, dt);
    return seed;
}

size_t weight_bias_zero_point_params::hash() const {
    size_t seed = weight_bias_params::hash();
    for (auto& dt : weights_zero_points)
        seed = hash_combine_dt(seed, dt);
    for (auto& dt : activations_zero_points)
        seed = hash_combine_dt(seed, dt);
    for (auto& dt : compensation)
        seed = hash_combine_dt(seed, dt);
    return seed;
}

}  // namespace kernel_selector
