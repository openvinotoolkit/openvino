// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_tools.hpp"

#include "runtime/pwl.h"

namespace ov {
namespace intel_gna {
namespace backend {

int64_t PWLTools::ComputeSlopeScale(const int32_t x_base) {
    return static_cast<int64_t>(1ULL << (8 * ((x_base & XBASE_SCALE_INDEX_MASK) + 1)));
}

int64_t PWLTools::ComputePWL(const gna_pwl_segment_t& segment, int64_t x) {
    auto segment_in_int64 = ConvertSegementTo64(segment);
    if (segment_in_int64.slope_scale == 0) {
        THROW_GNA_EXCEPTION << "slope_scale is 0, possible division by 0 when calculating function value";
    }
    return (x - segment_in_int64.x_base) * segment_in_int64.slope / segment_in_int64.slope_scale +
           segment_in_int64.y_base;
}

int64_t PWLTools::ComputeXForValuePWL(const gna_pwl_segment_t& segment, int64_t y) {
    auto segment_in_int64 = ConvertSegementTo64(segment);
    if (segment.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0, possible division by when calculation x for given y";
    }
    return segment_in_int64.x_base - segment_in_int64.y_base * segment_in_int64.slope_scale / segment_in_int64.slope;
}

PWLSegmentAs64 PWLTools::ConvertSegementTo64(const gna_pwl_segment_t& segment) {
    // conversion to int64_t done to avoid unitended converion to unsigned which cause errors
    const int64_t x_base = static_cast<int64_t>(static_cast<int32_t>(segment.xBase & XBASEMASK));
    const int64_t y_base = static_cast<int64_t>(segment.yBase);
    const int64_t slope = static_cast<int64_t>(segment.slope);
    const int64_t slope_scale = ComputeSlopeScale(segment.xBase);
    return {x_base, y_base, slope, slope_scale};
}

int64_t PWLTools::RoundTowardZero(const int64_t value) {
    //ensure that masking 2LSB will not affect final results
    return value / 4 * 4;
}

}  // namespace backend
}  // namespace intel_gna
}  // namespace ov