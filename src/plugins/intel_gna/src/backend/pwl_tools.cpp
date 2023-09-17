// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_tools.hpp"

#include "common/numerical_utils.hpp"
#include "gna_slope_scale.hpp"
#include "runtime/pwl.h"

namespace ov {
namespace intel_gna {
namespace backend {
namespace pwl_tools {

int32_t GetXBaseValue(const int32_t x_base) {
    return static_cast<int32_t>(x_base & XBASEMASK);
}

int32_t ComputeXBaseForSegment(const int32_t x_base, const int32_t index) {
    auto result = GetXBaseValue(x_base);
    return static_cast<int32_t>(result |= index);
}

int64_t ComputeSlopeScale(const int32_t x_base) {
    return static_cast<int64_t>(1ULL << (8 * (GetScaleIndex(x_base) + 1)));
}

PWLSegmentSlope ComputeSlopeForSegment(double slope, double in_scale, double out_scale) {
    const auto gna_slope_value = gna_slope(slope, in_scale, out_scale);
    auto segment_slope = common::DoubleToInt64(gna_slope_value.slope * gna_slope_value.slope_scale);

    if (segment_slope > std::numeric_limits<int16_t>::max()) {
        segment_slope = std::numeric_limits<int16_t>::max();
    }
    return {static_cast<int16_t>(segment_slope), static_cast<int16_t>(gna_slope_value.slope_scale_index)};
}

int32_t GetScaleIndex(const int32_t x_base) {
    return static_cast<int32_t>(x_base & XBASE_SCALE_INDEX_MASK);
}

int64_t ComputePWL(const gna_pwl_segment_t& segment, int64_t x) {
    auto segment_in_int64 = ConvertSegmentTo64(segment);
    if (segment_in_int64.slope_scale == 0) {
        THROW_GNA_EXCEPTION << "slope_scale is 0, possible division by 0 when calculating function value";
    }
    return (x - segment_in_int64.x_base) * segment_in_int64.slope / segment_in_int64.slope_scale +
           segment_in_int64.y_base;
}

int64_t ComputeXForValuePWL(const gna_pwl_segment_t& segment, int64_t y) {
    auto segment_in_int64 = ConvertSegmentTo64(segment);
    if (segment.slope == 0) {
        THROW_GNA_EXCEPTION << "Slope is 0, possible division by when calculation x for given y";
    }
    return segment_in_int64.x_base - segment_in_int64.y_base * segment_in_int64.slope_scale / segment_in_int64.slope;
}

PWLSegmentAs64 ConvertSegmentTo64(const gna_pwl_segment_t& segment) {
    // conversion to int64_t done to avoid unitended converion to unsigned which cause errors
    const int64_t x_base = GetXBaseValue(segment.xBase);
    const int64_t y_base = static_cast<int64_t>(segment.yBase);
    const int64_t slope = static_cast<int64_t>(segment.slope);
    const int64_t slope_scale = ComputeSlopeScale(segment.xBase);
    return {x_base, y_base, slope, slope_scale};
}

int64_t Round2LSBTowardZero(const int64_t value) {
    // ensure that masking 2LSB will not affect final results
    return value / 4 * 4;
}
}  // namespace pwl_tools
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov
