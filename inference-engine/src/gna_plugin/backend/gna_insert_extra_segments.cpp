// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "gna_insert_extra_segments.hpp"

void gna_insert_extra_segments(const std::vector<pwl_t> &pwl,
                               std::vector<gna_pwl_segment_t> &gna_pwl,
                               const double in_scale,
                               const double out_scale) {
    pwl_gna_slope_scale_t s;
    std::map<uint, gna_pwl_segment_t> extra_segments;
    gna_pwl_segment_t extra_segment;
    uint32_t gna_pwl_size = static_cast<int32_t>(pwl.size());
    for (uint32_t i = 2; i < gna_pwl_size - 1; i++) {
        if (pwl.empty()) break;
        s = gna_slope(pwl[i - 1].m, in_scale, out_scale);
        bool top_overflow = pwl[i - 1].beta * out_scale > INT16_MAX;
        bool down_overflow = pwl[i - 1].beta * out_scale < INT16_MIN;
        if (top_overflow) {
            extra_segment.xBase = static_cast<int32_t>(((INT16_MAX - pwl[i - 1].beta * out_scale) *
                    gna_pwl[i - 1].xBase - gna_pwl[i].xBase) /
                    ((pwl[i - 2].beta - pwl[i - 1].beta) * out_scale) + gna_pwl[i].xBase);
            extra_segment.xBase = extra_segment.xBase & XBASEMASK;
            extra_segment.xBase = extra_segment.xBase | s.slope_scale_index;
            extra_segment.yBase = INT16_MAX;
            extra_segment.slope = gna_pwl[i].slope;
            extra_segments[i - 1] = extra_segment;
            gna_pwl[i].yBase = INT16_MAX;
        }
        if (down_overflow) {
            // need insert extra segment on the right
            extra_segment.xBase = static_cast<int32_t>(((INT16_MAX - pwl[i - 1].beta * out_scale) *
                                                        gna_pwl[i + 1].xBase - gna_pwl[i].xBase) /
                                                       ((pwl[i].beta - pwl[i - 1].beta) * out_scale) + gna_pwl[i].xBase);
            extra_segment.xBase = extra_segment.xBase & XBASEMASK;
            extra_segment.xBase = extra_segment.xBase | s.slope_scale_index;
            extra_segment.yBase = INT16_MIN;
            extra_segment.slope = gna_pwl[i].slope;
            extra_segments[i] = extra_segment;
            gna_pwl[i].yBase = INT16_MIN;
        }
    }
    auto iterator = extra_segments.begin();
    for (; iterator != extra_segments.end(); iterator++) {
        gna_pwl.insert(gna_pwl.begin() + iterator->first, iterator->second);
    }
}