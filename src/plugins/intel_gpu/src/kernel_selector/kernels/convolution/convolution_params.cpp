// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_params.h"
#include "kernel_selector_utils.h"
#include <sstream>
#include <string>

namespace kernel_selector {
std::string convolution_params::to_string() const {
    std::stringstream s;

    s << parent::to_string() << "_";
    if (bias.empty()) {
        s << "no_bias"
          << "_";
    } else {
        s << "bias_" << bias[0].PhysicalSize() << "_";
    }
    s << filterSize.x << "_" << filterSize.y << "_";
    s << stride.x << "_" << stride.y << "_";
    s << dilation.x << "_" << dilation.y << "_";
    s << padding.x << "_" << padding.y << "_";
    s << split;

    return s.str();
}

std::string convolution_params::to_cache_string_v2() const {
    std::stringstream s;

    s << parent::to_cache_string_v2() << ";";
    s << filterSize.x << "_" << filterSize.y << "_" << filterSize.z << ";";
    s << stride.x << "_" << stride.y << "_" << stride.z << ";";
    s << dilation.x << "_" << dilation.y << "_" << dilation.z << ";";
    s << padding.x << "_" << padding.y << "_" << padding.z << ";";
    s << split << ";";
    s << groups;

    return s.str();
}

ParamsKey convolution_params::GetParamsKey() const {
    ParamsKey k = parent::GetParamsKey();

    if (split > 1) {
        k.EnableSplitSupport();
    }

    if (dilation.x != 1 || dilation.y != 1) {
        k.EnableDilation();
    }

    if (depthwise_separable_opt) {
        k.EnableDepthwiseSeparableOpt();
    }

    if (groups > 1 && !depthwise_separable_opt) {
        k.EnableGroupedConvolution();
    }

    if (deformable_mode) {
        k.EnableDeformableMode();
        if (bilinear_interpolation_pad)
            k.EnableBilinearInterpolationPad();
        if (deformable_mask_enabled)
            k.EnableDeformableMask();
    }

    k.EnableQuantization(quantization);

    return k;
}

size_t convolution_params::hash() const {
    auto seed = weight_bias_zero_point_params::hash();
    seed = hash_combine_usize(seed, filterSize);
    seed = hash_combine_usize(seed, stride);
    seed = hash_combine_usize(seed, dilation);
    seed = hash_combine_usize(seed, padding);

    seed = hash_combine(seed, split);
    seed = hash_combine(seed, depthwise_separable_opt);
    seed = hash_combine(seed, transposed);
    seed = hash_combine(seed, quantization);
    seed = hash_combine(seed, deformable_mode);
    seed = hash_combine(seed, groups);
    seed = hash_combine_usize(seed, kernelSize);
    seed = hash_combine(seed, deformable_groups);
    seed = hash_combine(seed, bilinear_interpolation_pad);
    seed = hash_combine(seed, deformable_mask_enabled);
    return seed;
}
}  // namespace kernel_selector
