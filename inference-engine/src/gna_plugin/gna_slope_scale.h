// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

typedef struct {
    double slope {};
    uint64_t slope_scale = 0;
    uint32_t slope_scale_index {};
} pwl_gna_slope_scale_t;

pwl_gna_slope_scale_t gna_slope(const double slope, const double in_scale, const double out_scale);
