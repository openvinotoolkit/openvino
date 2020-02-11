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
    levels(0), packed_binary_output(false), scale_shift_opt(false) {}

    int levels;
    bool packed_binary_output;
    bool scale_shift_opt;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        if (packed_binary_output)
            k.EnableQuantizePackedBinaryOutput();
        if (scale_shift_opt)
            k.EnableQuantizeScaleShiftOpt();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_optional_params : optional_params {
    quantize_optional_params() : optional_params(KernelType::QUANTIZE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// quantize_fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct quantize_fuse_params : fuse_params {
    quantize_fuse_params(bool scale_shift_opt, bool has_post_scale, bool has_post_shift)
    : fuse_params(KernelType::QUANTIZE)
    , scale_shift_opt(scale_shift_opt)
    , has_post_scale(has_post_scale)
    , has_post_shift(has_post_shift) { }

    bool scale_shift_opt;
    bool has_post_scale;
    bool has_post_shift;
};

}  // namespace kernel_selector
