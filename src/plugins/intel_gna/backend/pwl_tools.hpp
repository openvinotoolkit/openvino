// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend/gna_types.h"

namespace ov {
namespace intel_gna {
namespace backend {

struct PWLSegmentAs64 {
    int64_t x_base;
    int64_t y_base;
    int64_t slope;
    int64_t slope_scale;
};

class PWLTools {
public:
    /**
     * @brief Return slope_scale = 2^(3 * ( z + 1)) where z is the 2 less significant bits of xBase
     */
    static int64_t ComputeSlopeScale(const int32_t x_base);
    static int64_t ComputePWL(const gna_pwl_segment_t& segment, int64_t x);
    static int64_t ComputeXForValuePWL(const gna_pwl_segment_t& segment, int64_t y);
    static PWLSegmentAs64 ConvertSegementTo64(const gna_pwl_segment_t& segment);
    static int64_t RoundTowardZero(const int64_t value);
};
}  // namespace backend
}  // namespace intel_gna
}  // namespace ov