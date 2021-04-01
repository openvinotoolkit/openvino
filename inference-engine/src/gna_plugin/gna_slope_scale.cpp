// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <limits>

#include "gna_slope_scale.h"

pwl_gna_slope_scale_t gna_slope(const double slope,
                                const double in_scale,
                                const double out_scale) {
    pwl_gna_slope_scale_t s;
    s.slope = slope * out_scale / in_scale;

    for (s.slope_scale_index = 3; s.slope_scale_index > 0; --s.slope_scale_index) {
        s.slope_scale = static_cast<uint64_t>(1) << (8 * (1 + s.slope_scale_index));
        if (((s.slope * s.slope_scale) <= std::numeric_limits<int16_t>::max()) &&
                    ((s.slope * s.slope_scale) >= std::numeric_limits<int16_t>::min()))
            break;
    }
    s.slope_scale = static_cast<uint64_t>(1) << (8 * (1 + s.slope_scale_index));

    return(s);
}
