// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_params.h"
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
    s << padding_begin.x << "_" << padding_begin.y << "_";
    s << 1;

    return s.str();
}

std::string convolution_params::to_cache_string_v2() const {
    std::stringstream s;

    s << parent::to_cache_string_v2() << ";";
    s << filterSize.x << "_" << filterSize.y << "_" << filterSize.z << ";";
    s << stride.x << "_" << stride.y << "_" << stride.z << ";";
    s << dilation.x << "_" << dilation.y << "_" << dilation.z << ";";
    s << padding_begin.x << "_" << padding_begin.y << "_" << padding_begin.z << ";";
    s << 1 << ";";
    s << groups;

    return s.str();
}

ParamsKey convolution_params::GetParamsKey() const {
    ParamsKey k = parent::GetParamsKey();

    if (dilation.x != 1 || dilation.y != 1) {
        k.EnableDilation();
    }

    if (groups > 1) {
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
}  // namespace kernel_selector
