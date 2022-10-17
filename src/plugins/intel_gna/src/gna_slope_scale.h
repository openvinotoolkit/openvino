// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

struct pwl_gna_slope_scale_t {
    double slope {};
    uint64_t slope_scale = 0;
    uint32_t slope_scale_index {};
};

pwl_gna_slope_scale_t gna_slope(const double slope, const double in_scale, const double out_scale);
