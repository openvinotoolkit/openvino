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
    s << padding.x << "_" << padding.y << "_";
    s << split;

    return s.str();
}

std::string convolution_params::to_cache_string_v2() const {
    std::stringstream s;

    s << weight_bias_params::to_cache_string_v2() << ";";
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

    if (transposed) {
        k.EnableTranspose();
    }

    if (local_convolution) {
        k.EnableLocalConvolution();
    }

    if (groups > 1 && !depthwise_separable_opt) {
        k.EnableGroupedConvolution();
    }

    if (deformable_mode) {
        k.EnableDeformableMode();
    }

    k.EnableQuantization(quantization);

    return k;
}
}  // namespace kernel_selector
