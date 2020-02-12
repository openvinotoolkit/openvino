// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define PWL_FROM_FILE

#include <vector>
#include <iostream>

#include <runtime/pwl.h>
#include <gna_slope_scale.h>
#include "dnn_types.h"
#include "round_float_define.hpp"

void make_gna_pwl(const DnnActivation  fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  std::vector<intel_pwl_segment_t> &gna_pwl,
                  const uint32_t n) {
    pwl_gna_slope_scale_t s;
    uint32_t pwl_size = static_cast<int32_t>(pwl.size());
    gnalog() << "make_gna_pwl\n";
    gnalog() << "   in_scale  " << in_scale << "\n";
    gnalog() << "   out_scale " << out_scale << "\n";
    switch (fun) {
        case kActSigmoid:
        case kActTanh: {
            auto n_segments = static_cast<int32_t> (pwl_size) + 1;
            gna_pwl.resize(n_segments);
            // insert extra segment for x values < l_bound
            gna_pwl[0].xBase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
            if (fun == kActSigmoid) {
                gnalog() <<  "=========================== Sigmoid Segments ===========================\n";
                gna_pwl[0].yBase = gna_pwl[1].yBase = 0;
                gna_pwl[1].xBase = (static_cast<int32_t> (in_scale * (-pwl[0].b / pwl[0].m))) & XBASEMASK;
            } else {
                gnalog() <<  "=========================== Tanh Segments ===========================\n";
                gna_pwl[0].yBase = gna_pwl[1].yBase = static_cast<int16_t>(-1.0 * out_scale);
                gna_pwl[1].xBase = (static_cast<int32_t> (in_scale * (-1.0 - pwl[0].b) / pwl[0].m)) & XBASEMASK;
            }
            gna_pwl[0].slope = 0;

            gnalog() << (gna_pwl[0].xBase) / in_scale
                     << " " << (gna_pwl[0].yBase) / out_scale
                     << " " << 0.0
                     << "\n";

            s = gna_slope(pwl[0].m, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;

            gnalog() << ((int32_t)(gna_pwl[1].xBase & XBASEMASK) / in_scale)
                     << " " << (gna_pwl[1].yBase) / out_scale
                     << " " << pwl[0].m
                     << "\n";

            for (uint32_t i = 1; i < pwl_size - 1; ++i) {
                s = gna_slope(pwl[i].m, in_scale, out_scale);
                gna_pwl[i + 1].xBase = (static_cast<int32_t> (in_scale * pwl[i].alpha)) & XBASEMASK;
                gna_pwl[i + 1].yBase = FLOAT_TO_INT16(pwl[i].beta * out_scale);
                gna_pwl[i + 1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
                gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;

                gnalog() << (pwl[i].alpha)
                         << " " << pwl[i].beta
                         << " " << pwl[i].m
                         << "\n";
            }
            // insert extra segment for xvalues > u_bound
            gna_pwl[n_segments - 1].xBase =
                    ((uint32_t) (in_scale * (1.0 - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m)) & XBASEMASK;
            gna_pwl[n_segments - 1].yBase = FLOAT_TO_INT16(1.0 * out_scale);
            gna_pwl[n_segments - 1].slope = 0;

            gnalog() << (gna_pwl[n_segments - 1].xBase / in_scale)
                     << " " << 1.0
                     << " " << 0.0
                     << "\n";
            break;
        }
        case kActExp: {
            auto n_segments = static_cast<int32_t> (pwl_size) + 1;
            gna_pwl.resize(n_segments);
            // insert extra segment for x values < l_bound
            gna_pwl[0].xBase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
            gnalog() << "=========================== Exp Segments ===========================\n";
            gna_pwl[0].yBase = gna_pwl[1].yBase = 0;
            gna_pwl[1].xBase = (static_cast<int32_t> (in_scale * (-pwl[0].b / pwl[0].m))) & XBASEMASK;
            gna_pwl[0].slope = 0;

            gnalog() << (gna_pwl[0].xBase) / in_scale
                << " " << (gna_pwl[0].yBase) / out_scale
                << " " << 0.0
                << "\n";

            s = gna_slope(pwl[0].m, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;

            gnalog() << (gna_pwl[1].xBase / in_scale)
                << " " << (gna_pwl[1].yBase) / out_scale
                << " " << pwl[0].m
                << "\n";

            for (uint32_t i = 1; i < pwl_size - 1; ++i) {
                s = gna_slope(pwl[i].m, in_scale, out_scale);
                gna_pwl[i + 1].xBase = (static_cast<int32_t> (in_scale * pwl[i].alpha)) & XBASEMASK;
                gna_pwl[i + 1].yBase = FLOAT_TO_INT16(pwl[i].beta * out_scale);
                gna_pwl[i + 1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
                gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;

                gnalog() << (pwl[i].alpha)
                    << " " << pwl[i].beta
                    << " " << pwl[i].m
                    << "\n";
            }
            // insert extra segment for xvalues > u_bound
            gna_pwl[n_segments - 1].xBase =
                ((uint32_t)(in_scale * (INT16_MAX/out_scale - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m)) & XBASEMASK;
            gna_pwl[n_segments - 1].yBase = INT16_MAX;
            gna_pwl[n_segments - 1].slope = 0;

            gnalog() << (gna_pwl[n_segments - 1].xBase / in_scale)
                << " " << 1.0
                << " " << 0.0
                << "\n";
            break;
        }
        case kActLog: {
            auto n_segments = static_cast<int32_t> (pwl_size);
            gna_pwl.resize(n_segments);
            // insert extra segment for x values < l_bound
            gna_pwl[0].xBase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
            gnalog() << "=========================== Exp Segments ===========================\n";
            gna_pwl[0].yBase = gna_pwl[1].yBase = INT16_MIN;
            gna_pwl[1].xBase = (static_cast<int32_t> (1 + ~XBASEMASK));  // smallest representable value
            gna_pwl[0].slope = 0;

            gnalog() << (gna_pwl[0].xBase) / in_scale
                << " " << (gna_pwl[0].yBase) / out_scale
                << " " << 0.0
                << "\n";

            s = gna_slope(pwl[0].m, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;

            gnalog() << (gna_pwl[1].xBase / in_scale)
                << " " << (gna_pwl[1].yBase) / out_scale
                << " " << pwl[0].m
                << "\n";

            for (uint32_t i = 1; i < pwl_size - 1; ++i) {
                s = gna_slope(pwl[i].m, in_scale, out_scale);
                gna_pwl[i + 1].xBase = (static_cast<int32_t> (in_scale * pwl[i].alpha)) & XBASEMASK;
                gna_pwl[i + 1].yBase = FLOAT_TO_INT16(pwl[i].beta * out_scale);
                gna_pwl[i + 1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
                gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;

                gnalog() << (pwl[i].alpha)
                    << " " << pwl[i].beta
                    << " " << pwl[i].m
                    << "\n";
            }
            break;
        }
        case kActRelu:
        case kActLeakyRelu: {
            auto n_segments = 2;
            gna_pwl.resize(n_segments);

            gnalog() << "=========================== ReLU Segments ===========================\n";
            int32_t x_lower = INT32_MIN;
            int16_t y_lower = INT16_MIN;
            if (x_lower < y_lower * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
            if (y_lower < x_lower * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
            gna_pwl[0].yBase = y_lower * fun.negative_slope;
            s = gna_slope(fun.negative_slope, in_scale, out_scale);
            gna_pwl[0].xBase = (x_lower & XBASEMASK) | s.slope_scale_index;  // zero out the 2 lsb
            gna_pwl[0].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);

            gnalog() << gna_pwl[0].xBase / in_scale
                     << " " << gna_pwl[0].yBase / out_scale
                     << " " << (gna_pwl[0].slope * in_scale) / (out_scale*s.slope_scale)
                     << "\n";
            gna_pwl[1].xBase = 0;
            gna_pwl[1].yBase = 0;
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
            gnalog() << 0.0
                     << " " << 0.0
                     << " " << (gna_pwl[1].slope * in_scale) / (out_scale*s.slope_scale)
                     << "\n";
            break;
        }
        case kActIdentity:
        case kActKaldiLstmClipping:
        case kActDivByN: {
            int32_t x_lower = INT32_MIN;
            int32_t x_upper = INT32_MAX;
            int16_t y_lower = INT16_MIN;
            int16_t y_upper = INT16_MAX;
            auto n_segments = 2;
            if (fun == kActKaldiLstmClipping) {
                gnalog()  << "=========================== Clipping Segments ===========================\n";
                if (x_lower < l_bound * in_scale) {
                    if (y_lower < l_bound * out_scale) {
                        x_lower = FLOAT_TO_INT32(l_bound * in_scale);
                        y_lower = FLOAT_TO_INT16(l_bound * out_scale);
                    } else {
                        x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
                    }
                }
                if (x_upper > u_bound * in_scale) {
                    if (y_upper > u_bound * out_scale) {
                        x_upper = FLOAT_TO_INT32(u_bound * in_scale);
                        y_upper = FLOAT_TO_INT16(u_bound * out_scale);
                    } else {
                        x_upper = FLOAT_TO_INT32(y_upper  * in_scale / out_scale);
                    }
                }
            } else if (fun == kActIdentity) {
                gnalog() << "=========================== Identity Segments ===========================\n";
                if (x_lower < y_lower * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
                if (x_upper > y_upper * in_scale / out_scale) x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);
                if (y_lower < x_lower * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
                if (y_upper > x_upper * out_scale / in_scale) y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);
            } else {
                gnalog() << "=========================== DivByN Segments ===========================\n";
                if (x_lower < y_lower * (float)n * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * (float)n * in_scale / out_scale);
                if (x_upper > y_upper * (float)n * in_scale / out_scale) x_upper = FLOAT_TO_INT32(y_upper * (float)n * in_scale / out_scale);
                if (y_lower < x_lower * (1.0 / n) * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * (1.0 / n) * out_scale / in_scale);
                if (y_upper > x_upper* (1.0 / n) * out_scale / in_scale) y_upper = FLOAT_TO_INT16(x_upper * (1.0 / n) * out_scale / in_scale);
            }
 	    gna_pwl.resize(n_segments);
            gna_pwl[0].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
            gna_pwl[0].yBase = y_lower;
            gna_pwl[0].slope = 0;
            gnalog() << gna_pwl[0].xBase / in_scale
                     << " " << gna_pwl[0].yBase / out_scale
                     << " " << 0
                     << "\n";

            gna_pwl[1].xBase = x_lower & XBASEMASK;  // zero out the 2 lsb
            gna_pwl[1].yBase = y_lower;
            if (fun == kActDivByN) {
                s = gna_slope(1.0 / n, in_scale, out_scale);
            } else {
                s = gna_slope(1.0, in_scale, out_scale);
            }
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
            int32_t round_scale = FLOAT_TO_INT16(0.5f / s.slope) & XBASEMASK;
            gna_pwl[1].xBase = (gna_pwl[1].xBase - round_scale) | s.slope_scale_index;
            gnalog() << (int32_t)(gna_pwl[1].xBase & XBASEMASK) / in_scale
                    << " " << gna_pwl[1].yBase / out_scale
                    << " " << 1.0
                    << "\n";

            if (INT32_MAX > x_upper) {  // need a right segment
                gna_pwl.push_back({
                                          static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                                          y_upper,
                                          0 });

                gnalog() << (x_upper & XBASEMASK) / in_scale
                    << " " << gna_pwl[n_segments].yBase / out_scale
                    << " " << 0
                    << "\n";
            }
            break;
        }
        default:
            gnalog() << "Unexpected function activation!\n";
            std::cerr << "Unexpected function activation!\n";
    }
}
