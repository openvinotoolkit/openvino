// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <iostream>
#include <cmath>

#include <runtime/pwl.h>
#include <gna_slope_scale.h>
#include "dnn_types.h"
#include "backend/gna_types.h"
#include "round_float_define.hpp"

void make_gna_pwl(const DnnActivation  fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  std::vector<gna_pwl_segment_t> &gna_pwl) {
    pwl_gna_slope_scale_t s;
    uint32_t pwl_size = static_cast<int32_t>(pwl.size());
    gnalog() << "make_gna_pwl\n";
    gnalog() << "   in_scale  " << in_scale << "\n";
    gnalog() << "   out_scale " << out_scale << "\n";
    switch (fun) {
        case kActSigmoid:
        case kActTanh:
        case kActSoftSign: {
            auto n_segments = static_cast<int32_t> (pwl_size) + 1;
            gna_pwl.resize(n_segments);
            // insert extra segment for x values < l_bound
            gna_pwl[0].xBase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
            if (fun == kActSigmoid) {
                gnalog() <<  "=========================== Sigmoid Segments ===========================\n";
                gna_pwl[0].yBase = gna_pwl[1].yBase = 0;
                gna_pwl[1].xBase = (static_cast<int32_t> (in_scale * (-pwl[0].b / pwl[0].m))) & XBASEMASK;
            } else if (fun == kActTanh) {
                gnalog() <<  "=========================== Tanh Segments ===========================\n";
                gna_pwl[0].yBase = gna_pwl[1].yBase = static_cast<int16_t>(-1.0 * out_scale);
                gna_pwl[1].xBase = (static_cast<int32_t> (in_scale * (-1.0 - pwl[0].b) / pwl[0].m)) & XBASEMASK;
            } else {
                gnalog() << "=========================== SoftSign Segments ===========================\n";
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

            gnalog() << (gna_pwl[1].xBase/in_scale)
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
            gnalog() << "=========================== Log Segments ===========================\n";
            gna_pwl[0].yBase = gna_pwl[1].yBase = INT16_MIN;
            gna_pwl[1].xBase = (static_cast<int32_t> (1 + ~XBASEMASK));  // smallest representable value
            gna_pwl[0].slope = 0;

            gnalog() << gna_pwl[0].xBase / in_scale
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
            break;
        }
        case kActNegLog:
        case kActNegHalfLog: {
            auto n_segments = static_cast<int32_t> (pwl_size);
            gna_pwl.resize(n_segments);
            // insert extra segment for x values < l_bound
            gna_pwl[0].xBase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
            if (fun == kActNegHalfLog)
                gnalog() << "=========================== NegHalfLog Segments ===========================\n";
            else
                gnalog() << "=========================== NegLog Segments ===========================\n";
            gna_pwl[0].yBase = gna_pwl[1].yBase = INT16_MAX;
            gna_pwl[1].xBase = (static_cast<int32_t> (1 + ~XBASEMASK));  // smallest representable value
            gna_pwl[0].slope = 0;

            gnalog() << gna_pwl[0].xBase / in_scale
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
            break;
        }
        case kActRelu:
        case kActLeakyRelu: {
            auto n_segments = 2;
            gna_pwl.resize(n_segments);

            if (fun == kActRelu)
                gnalog() << "=========================== ReLU Segments ===========================\n";
            else
                gnalog() << "=========================== LeakyReLU Segments ======================\n";
            int32_t x_lower = INT32_MIN;
            int16_t y_lower = INT16_MIN;
            if (x_lower < y_lower * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
            if (y_lower < x_lower * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
            gna_pwl[0].yBase = y_lower * fun.args.lrelu.negative_slope;
            s = gna_slope(fun.args.lrelu.negative_slope, in_scale, out_scale);
            gna_pwl[0].xBase = (x_lower & XBASEMASK) | s.slope_scale_index;  // zero out the 2 lsb
            gna_pwl[0].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);

            gnalog() << (int32_t)(gna_pwl[0].xBase & XBASEMASK) / in_scale
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
        case kActSign: {
            auto n_segments = 3;
            gna_pwl.resize(n_segments);

            gnalog() << "=========================== Sign Segments ===========================\n";
            int32_t x_lower = INT32_MIN;
            int16_t y_lower = static_cast<int16_t>(-1.0 * out_scale);
            gna_pwl[0].yBase = y_lower;
            gna_pwl[0].xBase = (x_lower & XBASEMASK);  // zero out the 2 lsb
            gna_pwl[0].slope = 0;

            gnalog() << gna_pwl[0].xBase / in_scale
                << " " << gna_pwl[0].yBase / out_scale
                << " " << (gna_pwl[0].slope * in_scale) / (out_scale*s.slope_scale)
                << "\n";
            gna_pwl[1].xBase = -1;
            gna_pwl[1].yBase = 0;
            gna_pwl[1].slope = 0;
            gna_pwl[1].xBase = gna_pwl[1].xBase  & XBASEMASK;
            gnalog() << gna_pwl[1].xBase / in_scale
                << " " << gna_pwl[1].yBase / out_scale
                << " " << (gna_pwl[1].slope * in_scale) / (out_scale*s.slope_scale)
                << "\n";
            gna_pwl[2].xBase = 1 + ~XBASEMASK;  // smallest representable positive number
            gna_pwl[2].yBase = static_cast<int16_t>(1.0 * out_scale);
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[2].slope = 0;
            gna_pwl[2].xBase = gna_pwl[2].xBase & XBASEMASK;
            gnalog() << gna_pwl[2].xBase / in_scale
                << " " << gna_pwl[2].yBase / out_scale
                << " " << (gna_pwl[2].slope * in_scale) / (out_scale*s.slope_scale)
                << "\n";
            break;
        }
        case kActIdentity:
        case kActKaldiLstmClipping: {
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
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
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
        case kActAbs: {
            int32_t x_upper = INT32_MAX;
            int16_t y_upper = INT16_MAX;
            int32_t i = 0;

            auto n_segments = 2;

            if (y_upper > x_upper * out_scale / in_scale) y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);
            if (x_upper > y_upper * in_scale / out_scale) x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);

            gnalog() << "=========================== Abs Segments ===========================\n";
            if (y_upper == INT16_MAX) {  // saturation at ends - need one more segment
                n_segments += 1;
                gna_pwl.resize(n_segments);
                gna_pwl[i].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
                gna_pwl[i].yBase = INT16_MAX;
                gna_pwl[i].slope = 0;
                i++;
            } else {
                gna_pwl.resize(n_segments);
            }

            gna_pwl[i].xBase = (-x_upper) & XBASEMASK;  // zero out the 2 lsb
            gna_pwl[i].yBase = y_upper;
            s = gna_slope(-1.0, in_scale, out_scale);
            gna_pwl[i].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[i].xBase = gna_pwl[i].xBase | s.slope_scale_index;
            gnalog() << (int32_t)(gna_pwl[i].xBase & XBASEMASK) / in_scale
                << " " << gna_pwl[i].yBase / out_scale
                << " " << -1.0
                << "\n";
            gna_pwl[i + 1].xBase = 0;
            gna_pwl[i + 1].yBase = 0;
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[i + 1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;
            gnalog() << (int32_t)(gna_pwl[i + 1].xBase & XBASEMASK) / in_scale
                << " " << gna_pwl[i + 1].yBase / out_scale
                << " " << 1.0
                << "\n";
            break;
        }
        case kActPow: {
            float pow_exponent = fun.args.pow.exponent;
            if (pow_exponent == 0.0f || pow_exponent == 1.0f) {
                float pow_scale = fun.args.pow.scale;
                float pow_offset = fun.args.pow.offset;
                int32_t x_lower = INT32_MIN;
                int32_t x_upper = INT32_MAX;
                int16_t y_lower = INT16_MIN;
                int16_t y_upper = INT16_MAX;
                auto n_segments = 2;

                if (pow_exponent == 0.0f) {
                    y_lower = y_upper = FLOAT_TO_INT16(1 * out_scale);
                } else if (pow_exponent == 1.0f) {
                    if (x_lower < y_lower * in_scale / out_scale)
                        x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
                    if (x_upper > y_upper * in_scale / out_scale)
                        x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);
                    if (y_lower < x_lower * out_scale / in_scale)
                        y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
                    if (y_upper > x_upper * out_scale / in_scale)
                        y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);

                    if (pow_scale < 1) {
                        int16_t tmp = y_lower;
                        y_lower = y_upper;
                        y_upper = tmp;
                    }

                    int64_t x_lower_new = FLOAT_TO_INT32((x_lower / in_scale) / std::fabs(pow_scale) * in_scale);
                    int64_t x_upper_new = FLOAT_TO_INT32((x_upper / in_scale) / std::fabs(pow_scale) * in_scale);
                    x_lower = static_cast<int32_t>(x_lower_new);
                    x_upper = static_cast<int32_t>(x_upper_new);
                    if (x_lower_new < INT32_MIN) {
                        int16_t offset_lower = std::abs(x_lower_new - INT32_MIN) / in_scale * out_scale;
                        x_lower = INT32_MIN;
                        y_lower = y_lower + offset_lower;
                    }

                    if (x_upper_new > INT32_MAX) {
                        int16_t offset_upper = (x_upper_new - INT32_MAX) / in_scale * out_scale;
                        x_upper = INT32_MAX;
                        y_upper = y_upper - offset_upper;
                    }

                    int32_t y_lower_new = FLOAT_TO_INT32((y_lower / out_scale + pow_offset) * out_scale);
                    int32_t y_upper_new = FLOAT_TO_INT32((y_upper / out_scale + pow_offset) * out_scale);
                    y_lower = static_cast<int16_t>(y_lower_new);
                    y_upper = static_cast<int16_t>(y_upper_new);
                    if (y_lower_new < INT16_MIN) {
                        int32_t offset_lower = abs(y_lower_new - INT16_MIN) / out_scale * in_scale;
                        y_lower = INT16_MIN;
                        x_lower = x_lower + offset_lower;
                    }

                    if (y_lower_new > INT16_MAX) {
                        int32_t offset_lower = (y_lower_new - INT16_MAX) / out_scale * in_scale;
                        y_lower = INT16_MAX;
                        x_upper = x_upper + offset_lower;
                    }

                    if (y_upper_new > INT16_MAX) {
                        int32_t offset_upper = (y_upper_new - INT16_MAX) / out_scale * in_scale;
                        y_upper = INT16_MAX;
                        x_upper = x_upper - offset_upper;
                    }

                    if (y_upper_new < INT16_MIN) {
                        int32_t offset_upper = abs(y_upper_new - INT16_MAX) / out_scale * in_scale;
                        y_upper = INT16_MIN;
                        x_lower = x_lower - offset_upper;
                    }
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
                double slope = (static_cast<double>(y_upper - y_lower) / out_scale) / (static_cast<double>(x_upper - x_lower) / in_scale);
                s = gna_slope(slope, in_scale, out_scale);
                gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
                gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
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
                        << " " << gna_pwl[2].yBase / out_scale
                        << " " << 0
                        << "\n";
                }
            } else {
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
                    ((uint32_t)(in_scale * (INT16_MAX / out_scale - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m)) & XBASEMASK;
                gna_pwl[n_segments - 1].yBase = INT16_MAX;
                gna_pwl[n_segments - 1].slope = 0;

                gnalog() << (gna_pwl[n_segments - 1].xBase / in_scale)
                    << " " << 1.0
                    << " " << 0.0
                    << "\n";
                break;
            }
            break;
        }
        default:
            gnalog() << "Unexpected function activation!\n";
            std::cerr << "Unexpected function activation!\n";
    }
}
