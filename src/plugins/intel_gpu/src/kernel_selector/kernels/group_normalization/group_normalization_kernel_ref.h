// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GroupNormalizationParams
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct group_normalization_params : public base_params {
    group_normalization_params() : base_params(KernelType::GROUP_NORMALIZATION) {}

    std::int64_t num_groups{};
    double epsilon{};

    ParamsKey GetParamsKey() const override {
        return base_params::GetParamsKey();
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// group_normalization_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct group_normalization_optional_params : optional_params {
    group_normalization_optional_params() : optional_params(KernelType::GROUP_NORMALIZATION) {}
};
}  // namespace kernel_selector
