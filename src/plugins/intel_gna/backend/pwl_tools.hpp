// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"

namespace ov {
namespace intel_gna {
namespace backend {
namespace pwl_tools {

/**
 * Extract data from gna_pwl_segment_t and prepare in the form ready to be used in calculation.
 */
struct PWLSegmentAs64 {
    int64_t x_base;       // contains xBase & XBASEMASK
    int64_t y_base;       // yBase
    int64_t slope;        // slope
    int64_t slope_scale;  // 2^(3 * ( z + 1)) where z is the 2 less significant bits of xBase
};

struct PWLSegmentSlope {
    int16_t value;
    int32_t index;
};

int32_t GetXBaseValue(const int32_t x_base);
int32_t ComputeXBaseForSegment(const int32_t x_base, const int32_t index);
PWLSegmentSlope ComputeSlopeForSegment(double slope, double in_scale, double out_scale);
int64_t ComputeSlopeScale(const int32_t x_base);
int32_t GetScaleIndex(const int32_t x_base);
int64_t ComputePWL(const gna_pwl_segment_t& segment, int64_t x);
int64_t ComputeXForValuePWL(const gna_pwl_segment_t& segment, int64_t y);
PWLSegmentAs64 ConvertSegmentTo64(const gna_pwl_segment_t& segment);
int64_t Round2LSBTowardZero(const int64_t value);
}  // namespace pwl_tools
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov