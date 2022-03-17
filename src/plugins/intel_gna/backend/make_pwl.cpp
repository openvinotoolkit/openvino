// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "runtime/pwl.h"
#include "make_pwl.hpp"
#include "gna_slope_scale.h"
#include "dnn_types.h"
#include "backend/gna_types.h"
#include "round_float_define.hpp"


// This function performes emulatation of HW saturation of PWL segments in SW
// by inserting additional segments when overflow would happen
static void insert_extra_pwl_segments(std::vector<gna_pwl_segment_t>& gna_pwl,
    const int16_t y_min,
    const int16_t y_max) {
    std::map<size_t, gna_pwl_segment_t> extra_segments;
    gna_pwl_segment_t extra_segment;
    size_t gna_pwl_size = gna_pwl.size();

    if (gna_pwl_size == 0)
        return;

    // We're adding a segment at the beginning if the first one doesn't cover min value
    if ((gna_pwl[0].xBase & XBASEMASK) != (INT32_MIN & XBASEMASK)) {
        extra_segment.xBase = INT32_MIN & XBASEMASK;
        extra_segment.yBase = gna_pwl[0].yBase;
        extra_segment.slope = 0;
        extra_segments[0] = extra_segment;
    }

    // We're checking here if saturation could potentially happen at the trailing segments
    if (gna_pwl[gna_pwl_size - 1].slope != 0) {
        int16_t slope = gna_pwl[gna_pwl_size - 1].slope;
        int32_t xBase = gna_pwl[gna_pwl_size - 1].xBase & XBASEMASK;
        int16_t yBase = gna_pwl[gna_pwl_size - 1].yBase;
        float scale = pow(2, ((gna_pwl[gna_pwl_size - 1].xBase & ~XBASEMASK) + 1) * 8);
        float y_value = ((static_cast<float>(INT32_MAX) - xBase) * slope) / scale + yBase;

        if (y_value > static_cast<float>(INT16_MAX) || y_value < static_cast<float>(INT16_MIN)) {
            float x_value = ((static_cast<float>(y_max) - yBase) * scale) / slope + xBase;
            extra_segment.xBase = FLOAT_TO_INT32(x_value) & XBASEMASK;
            extra_segment.yBase = slope > 0 ? y_max : y_min;
            extra_segment.slope = 0;
            extra_segments[gna_pwl_size] = extra_segment;
        }
    }

    if (!extra_segments.empty())
        gnalog() << "Additional segment(s) added to protect against saturation\n";

    for (auto i = extra_segments.rbegin(); i != extra_segments.rend(); i++) {
        gna_pwl.insert(gna_pwl.begin() + i->first, i->second);
    }
}

static void print_segments_header(const DnnActivation&  fun) {
    gnalog() <<  "=========================== " << intel_dnn_activation_name[fun] <<
                 " segments ===========================\n";
    gnalog() << std::setw(12) << std::setfill(' ') << "x" << std::setw(12) << std::setfill(' ') <<
                "y" << std::setw(12) << std::setfill(' ') << "slope" << std::endl;
}

static void print_segment(double x, double y, double slope) {
    gnalog() << std::setw(12) << std::setfill(' ') << x << std::setw(12) << std::setfill(' ') <<
                y << std::setw(12) << std::setfill(' ') << slope << std::endl;
}

static std::vector<gna_pwl_segment_t> create_multisegment_gna_pwl(const std::vector<pwl_t>& pwl,
                                                                  double in_scale,
                                                                  double out_scale,
                                                                  double min_x_val,
                                                                  double max_x_val,
                                                                  double min_y_val,
                                                                  double max_y_val,
                                                                  bool fake_quantize,
                                                                  bool add_last_seg) {
    std::vector<gna_pwl_segment_t> gna_pwl;

    int32_t xbase = static_cast<int32_t> (INT32_MIN & XBASEMASK);  // zero out the 2 lsb
    int16_t ybase = FLOAT_TO_INT16(min_y_val * out_scale);
    int16_t slope = 0;
    gna_pwl.push_back({xbase, ybase, slope});
    print_segment(xbase / in_scale, min_y_val, slope);

    if (!fake_quantize && min_x_val > INT32_MIN / in_scale) {
        auto s = gna_slope(pwl[0].m, in_scale, out_scale);
        slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
        xbase = (static_cast<int32_t>(min_x_val * in_scale) & XBASEMASK) | s.slope_scale_index;
        ybase = FLOAT_TO_INT16(min_y_val * out_scale);
        gna_pwl.push_back({xbase, ybase, slope});
        print_segment(min_x_val, min_y_val, pwl[0].m);
    }

    for (uint32_t i = 0; i < pwl.size(); ++i) {
        if (!fake_quantize && (pwl[i].alpha <= min_x_val ||
            pwl[i].alpha <= INT32_MIN / in_scale ||
            pwl[i].alpha >= max_x_val)) {
            continue;
        }

        auto s = gna_slope(pwl[i].m, in_scale, out_scale);
        xbase = ((static_cast<int32_t> (in_scale * pwl[i].alpha)) & XBASEMASK) | s.slope_scale_index;
        ybase = FLOAT_TO_INT16(pwl[i].beta * out_scale);
        slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
        gna_pwl.push_back({xbase, ybase, slope});
        print_segment(pwl[i].alpha, pwl[i].beta, pwl[i].m);
    }

    if (!fake_quantize && add_last_seg) {
        // insert extra segment for xvalues > u_bound
        xbase = static_cast<int32_t>(max_x_val * in_scale) & XBASEMASK;
        ybase = FLOAT_TO_INT16(max_y_val * out_scale);
        slope = 0;
        gna_pwl.push_back({xbase, ybase, slope});
        print_segment(max_x_val, max_y_val, slope);
    }

    return gna_pwl;
}

void make_gna_pwl(const DnnActivation&  fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  const bool low_precision,
                  std::vector<gna_pwl_segment_t> &gna_pwl) {
    pwl_gna_slope_scale_t s;
    const int16_t y_min = low_precision ? INT8_MIN : INT16_MIN;
    const int16_t y_max = low_precision ? INT8_MAX : INT16_MAX;
    uint32_t pwl_size = static_cast<int32_t>(pwl.size());
    gnalog() << "make_gna_pwl\n";
    gnalog() << "   in_scale  " << in_scale << "\n";
    gnalog() << "   out_scale " << out_scale << "\n";
    print_segments_header(fun);
    switch (fun) {
        case kActSigmoid:
        case kActTanh:
        case kActSoftSign: {
            // insert extra segment for x values < l_bound
            double min_x_val;
            double min_y_val;
            if (fun == kActSigmoid) {
                min_y_val = fun.fqParams.set ? pwl[0].beta : 0;
                min_x_val = -pwl[0].b / pwl[0].m;
            } else if (fun == kActTanh) {
                min_y_val = fun.fqParams.set ? pwl[0].beta : -1.0;
                min_x_val = (-1.0 - pwl[0].b) / pwl[0].m;
            } else {
                min_y_val = fun.fqParams.set ? pwl[0].beta : -1.0;
                min_x_val = (-1.0 - pwl[0].b) / pwl[0].m;
            }
            double max_y_val = fun.fqParams.set ? pwl.back().beta : 1.0;
            double max_x_val = fun.srcFQParams.set ? u_bound : (1.0 - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m;
            gna_pwl = create_multisegment_gna_pwl(pwl, in_scale, out_scale, min_x_val, max_x_val, min_y_val, max_y_val,
                fun.fqParams.set, true);
            break;
        }
        case kActExp: {
            double min_x_val = -pwl[0].b / pwl[0].m;
            double max_x_val = (y_max/out_scale - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m;
            double min_y_val = fun.fqParams.set ? pwl[0].beta : 0;
            double max_y_val = fun.fqParams.set ? pwl.front().beta : y_max / out_scale;
            gna_pwl = create_multisegment_gna_pwl(pwl, in_scale, out_scale, min_x_val, max_x_val, min_y_val, max_y_val,
                fun.fqParams.set, true);
            break;
        }
        case kActLog: {
            double min_x_val = (1 + ~XBASEMASK) / in_scale;
            double max_x_val = INT32_MAX / in_scale;
            double min_y_val = y_min / out_scale;
            double max_y_val = y_max / out_scale;
            gna_pwl = create_multisegment_gna_pwl(pwl, in_scale, out_scale, min_x_val, max_x_val, min_y_val, max_y_val,
                fun.fqParams.set, false);
            break;
        }
        case kActNegLog:
        case kActNegHalfLog: {
            double min_x_val = 1 + ~XBASEMASK;
            double max_x_val = INT32_MAX / in_scale;
            double min_y_val = y_max / out_scale;
            double max_y_val = y_min / out_scale;
            gna_pwl = create_multisegment_gna_pwl(pwl, in_scale, out_scale, min_x_val, max_x_val, min_y_val, max_y_val,
                fun.fqParams.set, false);
            break;
        }
        case kActRelu:
        case kActLeakyRelu: {
            auto n_segments = 2;
            gna_pwl.resize(n_segments);

            int32_t x_lower = INT32_MIN;
            int32_t x_upper = INT32_MAX;
            int32_t y_lower = y_min;
            int16_t y_upper = y_max;
            if (fun.fqParams.set) {
                x_lower = std::max(FLOAT_TO_INT64(*fun.fqParams.input_low * 1.25 * in_scale), static_cast<int64_t>(x_lower));
                x_upper = std::min(FLOAT_TO_INT64(*fun.fqParams.input_high * 1.25 * in_scale), static_cast<int64_t>(x_upper));
                // y_lower can be reduced with negative slope
                y_lower = *fun.fqParams.input_low * 1.25 * out_scale;
                y_upper = std::min(FLOAT_TO_INT32(*fun.fqParams.input_high * 1.25 * out_scale), static_cast<int32_t>(y_upper));
            } else {
                if (x_lower < y_lower * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
                if (y_lower < x_lower * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
            }

            gna_pwl[0].yBase = std::max(FLOAT_TO_INT32(y_lower * fun.args.lrelu.negative_slope), static_cast<int32_t>(y_min));
            s = gna_slope(fun.args.lrelu.negative_slope, in_scale, out_scale);
            gna_pwl[0].xBase = (x_lower & XBASEMASK) | s.slope_scale_index;  // zero out the 2 lsb
            gna_pwl[0].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);

            print_segment((int32_t)(gna_pwl[0].xBase & XBASEMASK) / in_scale,
                          gna_pwl[0].yBase / out_scale,
                          (gna_pwl[0].slope * in_scale) / (out_scale*s.slope_scale));

            gna_pwl[1].xBase = 0;
            gna_pwl[1].yBase = 0;
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
            print_segment(0.0, 0.0, (gna_pwl[1].slope * in_scale) / (out_scale*s.slope_scale));

            if (fun.fqParams.set) {  // need a right segment
                gna_pwl.push_back({
                    static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                    y_upper,
                    0 });

                print_segment((x_upper & XBASEMASK) / in_scale, gna_pwl[n_segments].yBase / out_scale, 0.0);
            }
            break;
        }
        case kActSign: {
            auto n_segments = 3;
            gna_pwl.resize(n_segments);

            int32_t x_lower = INT32_MIN;
            int16_t y_lower = static_cast<int16_t>(-1.0 * out_scale);
            gna_pwl[0].yBase = y_lower;
            gna_pwl[0].xBase = (x_lower & XBASEMASK);  // zero out the 2 lsb
            gna_pwl[0].slope = 0;

            print_segment(gna_pwl[0].xBase / in_scale, gna_pwl[0].yBase / out_scale,
                (gna_pwl[0].slope * in_scale) / (out_scale*s.slope_scale));
            gna_pwl[1].xBase = -1;
            gna_pwl[1].yBase = 0;
            gna_pwl[1].slope = 0;
            gna_pwl[1].xBase = gna_pwl[1].xBase  & XBASEMASK;
            print_segment(gna_pwl[1].xBase / in_scale, gna_pwl[1].yBase / out_scale,
                (gna_pwl[1].slope * in_scale) / (out_scale*s.slope_scale));

            gna_pwl[2].xBase = 1 + ~XBASEMASK;  // smallest representable positive number
            gna_pwl[2].yBase = static_cast<int16_t>(1.0 * out_scale);
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[2].slope = 0;
            gna_pwl[2].xBase = gna_pwl[2].xBase & XBASEMASK;
            print_segment(gna_pwl[2].xBase / in_scale, gna_pwl[2].yBase / out_scale,
                (gna_pwl[2].slope * in_scale) / (out_scale*s.slope_scale));
            break;
        }
        case kActIdentity:
        case kActKaldiLstmClipping:
        case kActFakeQuantize: {
            int32_t x_lower = INT32_MIN;
            int32_t x_upper = INT32_MAX;
            int16_t y_lower = y_min;
            int16_t y_upper = y_max;
            if (fun == kActFakeQuantize && fun.fqParams.set) {
                x_lower = std::max(static_cast<int64_t>(*fun.fqParams.input_low * in_scale), static_cast<int64_t>(x_lower));
                x_upper = std::min(static_cast<int64_t>(*fun.fqParams.input_high * in_scale), static_cast<int64_t>(x_upper));
                y_lower = std::max(static_cast<int32_t>(*fun.fqParams.input_low * out_scale), static_cast<int32_t>(y_lower));
                y_upper = std::min(static_cast<int32_t>(*fun.fqParams.input_high * out_scale), static_cast<int32_t>(y_upper));
            }
            auto n_segments = 2;
            if (fun == kActKaldiLstmClipping) {
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
                if (x_lower < y_lower * in_scale / out_scale) x_lower = FLOAT_TO_INT32(y_lower * in_scale / out_scale);
                if (x_upper > y_upper * in_scale / out_scale) x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);
                if (y_lower < x_lower * out_scale / in_scale) y_lower = FLOAT_TO_INT16(x_lower * out_scale / in_scale);
                if (y_upper > x_upper * out_scale / in_scale) y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);
            }

            gna_pwl.resize(n_segments);
            gna_pwl[0].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
            gna_pwl[0].yBase = y_lower;
            gna_pwl[0].slope = 0;
            print_segment(gna_pwl[0].xBase / in_scale, gna_pwl[0].yBase / out_scale, 0.0);

            gna_pwl[1].xBase = x_lower & XBASEMASK;  // zero out the 2 lsb
            gna_pwl[1].yBase = y_lower;
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
            print_segment((int32_t)(gna_pwl[1].xBase & XBASEMASK) / in_scale, gna_pwl[1].yBase / out_scale, 1.0);

            if (INT32_MAX > x_upper) {  // need a right segment
                gna_pwl.push_back({
                    static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                    y_upper,
                    0 });

                print_segment((x_upper & XBASEMASK) / in_scale, gna_pwl[n_segments].yBase / out_scale, 0.0);
            }
            break;
        }
        case kActAbs: {
            int32_t x_upper = INT32_MAX;
            int16_t y_upper = y_max;
            int32_t i = 0;

            auto n_segments = 2;

            if (y_upper > x_upper * out_scale / in_scale) y_upper = FLOAT_TO_INT16(x_upper * out_scale / in_scale);
            if (x_upper > y_upper * in_scale / out_scale) x_upper = FLOAT_TO_INT32(y_upper * in_scale / out_scale);

            if (y_upper == y_max) {  // saturation at ends - need one more segment
                n_segments += 1;
                gna_pwl.resize(n_segments);
                gna_pwl[i].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
                gna_pwl[i].yBase = y_max;
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
            print_segment((int32_t)(gna_pwl[i].xBase & XBASEMASK) / in_scale, gna_pwl[i].yBase / out_scale, -1.0);

            gna_pwl[i + 1].xBase = 0;
            gna_pwl[i + 1].yBase = 0;
            s = gna_slope(1.0, in_scale, out_scale);
            gna_pwl[i + 1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;
            print_segment((int32_t)(gna_pwl[i + 1].xBase & XBASEMASK) / in_scale, gna_pwl[i + 1].yBase / out_scale, 1.0);
            break;
        }
        case kActPow: {
            float pow_exponent = fun.args.pow.exponent;
            IE_ASSERT(pow_exponent != 1.0f);
            if (pow_exponent == 0.0f) {
                int32_t x_lower = INT32_MIN;
                int32_t x_upper = INT32_MAX;
                int16_t y_lower = FLOAT_TO_INT16(1 * out_scale);
                int16_t y_upper = y_lower;
                auto n_segments = 2;

                gna_pwl.resize(n_segments);

                gna_pwl[0].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
                gna_pwl[0].yBase = y_lower;
                gna_pwl[0].slope = 0;
                print_segment(gna_pwl[0].xBase / in_scale, gna_pwl[0].yBase / out_scale, 0.0);

                gna_pwl[1].xBase = x_lower & XBASEMASK;  // zero out the 2 lsb
                gna_pwl[1].yBase = y_lower;
                double slope = (static_cast<double>(y_upper - y_lower) / out_scale) / (static_cast<double>(x_upper - x_lower) / in_scale);
                s = gna_slope(slope, in_scale, out_scale);
                gna_pwl[1].slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
                gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
                print_segment((int32_t)(gna_pwl[1].xBase & XBASEMASK) / in_scale, gna_pwl[1].yBase / out_scale, 1.0);
            } else {
                double min_x_val = -pwl[0].b / pwl[0].m;
                double max_x_val = (y_max/out_scale - pwl[pwl_size - 2].b) / pwl[pwl_size - 2].m;
                double min_y_val = fun.fqParams.set ? pwl[0].beta : 0;
                double max_y_val = fun.fqParams.set ? pwl.front().beta : y_max / out_scale;
                gna_pwl = create_multisegment_gna_pwl(pwl, in_scale, out_scale, min_x_val, max_x_val, min_y_val, max_y_val,
                    fun.fqParams.set, true);
                break;
            }
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Unexpected function activation!" << fun;
    }
    insert_extra_pwl_segments(gna_pwl, y_min, y_max);
}
