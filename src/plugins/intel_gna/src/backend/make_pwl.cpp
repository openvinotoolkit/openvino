// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "make_pwl.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <ngraph/op/constant.hpp>
#include <vector>

#include "backend/gna_types.hpp"
#include "common/numerical_utils.hpp"
#include "dnn_types.hpp"
#include "gna_slope_scale.hpp"
#include "log/log.hpp"
#include "pwl_input_params.hpp"
#include "pwl_segments_creator_factory.hpp"
#include "runtime/pwl.h"

using namespace ov::intel_gna;
using namespace ov::intel_gna::common;

// This function performs emulation of HW saturation of PWL segments in SW
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
        float scale = static_cast<float>(pow(2, ((gna_pwl[gna_pwl_size - 1].xBase & ~XBASEMASK) + 1) * 8));
        float y_value = ((static_cast<float>(INT32_MAX) - xBase) * slope) / scale + yBase;

        if (y_value > static_cast<float>(INT16_MAX) || y_value < static_cast<float>(INT16_MIN)) {
            float x_value = ((static_cast<float>(y_max) - yBase) * scale) / slope + xBase;
            extra_segment.xBase = FloatToInt32(x_value) & XBASEMASK;
            extra_segment.yBase = slope > 0 ? y_max : y_min;
            extra_segment.slope = 0;
            extra_segments[gna_pwl_size] = extra_segment;
        }
    }

    if (!extra_segments.empty())
        log::debug() << "Additional segment(s) added to protect against saturation\n";

    for (auto i = extra_segments.rbegin(); i != extra_segments.rend(); i++) {
        gna_pwl.insert(gna_pwl.begin() + i->first, i->second);
    }
}

static void print_segments_header(const DnnActivation& fun) {
    log::debug() << "=========================== " << intel_dnn_activation_name[fun]
                 << " segments ===========================\n";
    log::debug() << std::setw(12) << std::setfill(' ') << "x" << std::setw(12) << std::setfill(' ') << "y"
                 << std::setw(12) << std::setfill(' ') << "slope" << std::endl;
}

static void print_segments_header() {
    log::debug() << "=========================== segments ===========================\n";
    log::debug() << std::setw(12) << std::setfill(' ') << "x" << std::setw(12) << std::setfill(' ') << "y"
                 << std::setw(12) << std::setfill(' ') << "slope" << std::endl;
}

static void print_segment(double x, double y, double slope) {
    log::debug() << std::setw(12) << std::setfill(' ') << x << std::setw(12) << std::setfill(' ') << y << std::setw(12)
                 << std::setfill(' ') << slope << std::endl;
}

void make_gna_pwl(const DnnActivation& fun,
                  const std::vector<pwl_t>& pwl,
                  const double l_bound,
                  const double u_bound,
                  const double in_scale,
                  const double out_scale,
                  const bool low_precision,
                  const bool is_fused_with_conv2d,
                  std::vector<gna_pwl_segment_t>& gna_pwl) {
    log::debug() << "make_gna_pwl\n";
    log::debug() << "   in_scale  " << in_scale << "\n";
    log::debug() << "   out_scale " << out_scale << "\n";
    print_segments_header(fun);
    if (fun.type == kActIdentity) {
        auto pwl_creator = ov::intel_gna::backend::PWLSegmentsCreatorFactory::CreateCreator(fun.type);
        if (pwl_creator == nullptr) {
            THROW_GNA_EXCEPTION << "Could not find creator for function type equal: " << static_cast<int>(fun.type);
        }
        ov::intel_gna::backend::PWLInputParams input_data(low_precision, fun.fqParams, in_scale, out_scale);
        auto segments_with_borders = pwl_creator->CreateSegmentsWithBorders(input_data);
        gna_pwl = segments_with_borders.segments;
        auto& y_min_max = segments_with_borders.border_values.y_min_max;
        // Extra segments are needed only in case activation function was fused with Conv2D layer
        if (is_fused_with_conv2d) {
            insert_extra_pwl_segments(gna_pwl, y_min_max.y_min, y_min_max.y_max);
        }
        return;
    }

    pwl_gna_slope_scale_t s;
    const int16_t y_min = low_precision ? INT8_MIN : INT16_MIN;
    const int16_t y_max = low_precision ? INT8_MAX : INT16_MAX;
    switch (fun) {
    case kActRelu:
    case kActLeakyRelu: {
        auto n_segments = 2;
        gna_pwl.resize(n_segments);
        int32_t x_lower = INT32_MIN;
        int32_t x_upper = INT32_MAX;
        int32_t y_lower = y_min;
        int16_t y_upper = y_max;
        if (fun.fqParams.set) {
            x_lower = static_cast<int32_t>(
                std::max(DoubleToInt64(*fun.fqParams.input_low * 1.25 * in_scale), static_cast<int64_t>(x_lower)));
            x_upper = static_cast<int32_t>(
                std::min(DoubleToInt64(*fun.fqParams.input_high * 1.25 * in_scale), static_cast<int64_t>(x_upper)));
            // y_lower can be reduced with negative slope
            y_lower = static_cast<int32_t>(*fun.fqParams.input_low * 1.25 * out_scale);
            y_upper = static_cast<int16_t>(
                std::min(DoubleToInt32(*fun.fqParams.input_high * 1.25 * out_scale), static_cast<int32_t>(y_upper)));
        } else {
            if (x_lower < y_lower * in_scale / out_scale)
                x_lower = DoubleToInt32(y_lower * in_scale / out_scale);
            if (y_lower < x_lower * out_scale / in_scale)
                y_lower = DoubleToInt16(x_lower * out_scale / in_scale);
        }

        gna_pwl[0].yBase = std::max(FloatToInt32(y_lower * fun.args.lrelu.negative_slope), static_cast<int32_t>(y_min));
        s = gna_slope(fun.args.lrelu.negative_slope, in_scale, out_scale);
        gna_pwl[0].xBase = (x_lower & XBASEMASK) | s.slope_scale_index;  // zero out the 2 lsb
        gna_pwl[0].slope = DoubleToInt16(s.slope * s.slope_scale);

        print_segment((int32_t)(gna_pwl[0].xBase & XBASEMASK) / in_scale,
                      gna_pwl[0].yBase / out_scale,
                      (gna_pwl[0].slope * in_scale) / (out_scale * s.slope_scale));

        gna_pwl[1].xBase = 0;
        gna_pwl[1].yBase = 0;
        s = gna_slope(1.0, in_scale, out_scale);
        gna_pwl[1].slope = DoubleToInt16(s.slope * s.slope_scale);
        gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
        print_segment(0.0, 0.0, (gna_pwl[1].slope * in_scale) / (out_scale * s.slope_scale));

        if (fun.fqParams.set) {                                            // need a right segment
            gna_pwl.push_back({static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                               y_upper,
                               0});

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

        print_segment(gna_pwl[0].xBase / in_scale,
                      gna_pwl[0].yBase / out_scale,
                      (gna_pwl[0].slope * in_scale) / (out_scale * s.slope_scale));
        gna_pwl[1].xBase = -1;
        gna_pwl[1].yBase = 0;
        gna_pwl[1].slope = 0;
        gna_pwl[1].xBase = gna_pwl[1].xBase & XBASEMASK;
        print_segment(gna_pwl[1].xBase / in_scale,
                      gna_pwl[1].yBase / out_scale,
                      (gna_pwl[1].slope * in_scale) / (out_scale * s.slope_scale));

        gna_pwl[2].xBase = 1 + ~XBASEMASK;  // smallest representable positive number
        gna_pwl[2].yBase = static_cast<int16_t>(1.0 * out_scale);
        s = gna_slope(1.0, in_scale, out_scale);
        gna_pwl[2].slope = 0;
        gna_pwl[2].xBase = gna_pwl[2].xBase & XBASEMASK;
        print_segment(gna_pwl[2].xBase / in_scale,
                      gna_pwl[2].yBase / out_scale,
                      (gna_pwl[2].slope * in_scale) / (out_scale * s.slope_scale));
        break;
    }
    case kActKaldiLstmClipping:
    case kActFakeQuantize: {
        int32_t x_lower = INT32_MIN;
        int32_t x_upper = INT32_MAX;
        int16_t y_lower = y_min;
        int16_t y_upper = y_max;
        if (fun.fqParams.set) {
            x_lower = static_cast<int32_t>(
                std::max(static_cast<int64_t>(*fun.fqParams.input_low * in_scale), static_cast<int64_t>(x_lower)));
            x_upper = static_cast<int32_t>(
                std::min(static_cast<int64_t>(*fun.fqParams.input_high * in_scale), static_cast<int64_t>(x_upper)));
            y_lower =
                std::max(static_cast<int32_t>(*fun.fqParams.input_low * out_scale), static_cast<int32_t>(y_lower));
            y_upper =
                std::min(static_cast<int32_t>(*fun.fqParams.input_high * out_scale), static_cast<int32_t>(y_upper));
        }
        auto n_segments = 2;
        if (fun == kActKaldiLstmClipping) {
            if (x_lower < l_bound * in_scale) {
                if (y_lower < l_bound * out_scale) {
                    x_lower = DoubleToInt32(l_bound * in_scale);
                    y_lower = DoubleToInt16(l_bound * out_scale);
                } else {
                    x_lower = DoubleToInt32(y_lower * in_scale / out_scale);
                }
            }
            if (x_upper > u_bound * in_scale) {
                if (y_upper > u_bound * out_scale) {
                    x_upper = DoubleToInt32(u_bound * in_scale);
                    y_upper = DoubleToInt16(u_bound * out_scale);
                } else {
                    x_upper = DoubleToInt32(y_upper * in_scale / out_scale);
                }
            }
        }

        gna_pwl.resize(n_segments);
        gna_pwl[0].xBase = INT32_MIN & XBASEMASK;  // zero out the 2 lsb
        gna_pwl[0].yBase = y_lower;
        gna_pwl[0].slope = 0;
        print_segment(gna_pwl[0].xBase / in_scale, gna_pwl[0].yBase / out_scale, 0.0);

        gna_pwl[1].xBase = x_lower & XBASEMASK;  // zero out the 2 lsb
        gna_pwl[1].yBase = y_lower;
        s = gna_slope(1.0, in_scale, out_scale);
        gna_pwl[1].slope = DoubleToInt16(s.slope * s.slope_scale);
        gna_pwl[1].xBase = gna_pwl[1].xBase | s.slope_scale_index;
        print_segment((int32_t)(gna_pwl[1].xBase & XBASEMASK) / in_scale, gna_pwl[1].yBase / out_scale, 1.0);

        if (INT32_MAX > x_upper) {                                         // need a right segment
            gna_pwl.push_back({static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                               y_upper,
                               0});

            print_segment((x_upper & XBASEMASK) / in_scale, gna_pwl[n_segments].yBase / out_scale, 0.0);
        }
        break;
    }
    case kActAbs: {
        int32_t x_upper = INT32_MAX;
        int16_t y_upper = y_max;
        int32_t i = 0;

        auto n_segments = 2;

        if (y_upper > x_upper * out_scale / in_scale)
            y_upper = DoubleToInt16(x_upper * out_scale / in_scale);
        if (x_upper > y_upper * in_scale / out_scale)
            x_upper = DoubleToInt32(y_upper * in_scale / out_scale);

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
        gna_pwl[i].slope = DoubleToInt16(s.slope * s.slope_scale);
        gna_pwl[i].xBase = gna_pwl[i].xBase | s.slope_scale_index;
        print_segment((int32_t)(gna_pwl[i].xBase & XBASEMASK) / in_scale, gna_pwl[i].yBase / out_scale, -1.0);

        gna_pwl[i + 1].xBase = 0;
        gna_pwl[i + 1].yBase = 0;
        s = gna_slope(1.0, in_scale, out_scale);
        gna_pwl[i + 1].slope = DoubleToInt16(s.slope * s.slope_scale);
        gna_pwl[i + 1].xBase = gna_pwl[i + 1].xBase | s.slope_scale_index;
        print_segment((int32_t)(gna_pwl[i + 1].xBase & XBASEMASK) / in_scale, gna_pwl[i + 1].yBase / out_scale, 1.0);
        break;
    }
    default:
        THROW_GNA_EXCEPTION << "Unexpected function activation!" << fun;
    }

    // Extra segments are needed only in case activation function was fused with Conv2D layer
    if (is_fused_with_conv2d) {
        insert_extra_pwl_segments(gna_pwl, y_min, y_max);
    }
}

template <typename T>
static T cast_check_overflow(double v, bool round = true) {
    if (v == std::numeric_limits<double>::infinity() || v > std::numeric_limits<T>::max()) {
        return std::numeric_limits<T>::max();
    }

    if (v == -std::numeric_limits<double>::infinity() || v < std::numeric_limits<T>::min()) {
        return std::numeric_limits<T>::min();
    }

    return round ? DoubleToInt32(v) : static_cast<T>(v);
}

/**
 * Make GNA segments from PWL (m*x + b).
 * @param m - array of slopes of a function
 * @param b - array of offset of a function
 * @param alpha - array x-the value of the beginning of a segments
 * @param count - number of PWL segments
 * @param in_scale - input scale factor
 * @param out_scale - output scale factor
 */
template <typename T>
static void make_gna_pwl(const T* m,
                         const T* b,
                         const T* alpha,
                         size_t count,
                         double in_scale,
                         double out_scale,
                         std::vector<gna_pwl_segment_t>& gna_pwl) {
    auto mul_check_overflow = [](double a, double b) -> double {
        if (a == 0 || b == 0) {
            return 0;
        }

        if (std::fabs(a) == std::numeric_limits<double>::infinity() ||
            std::fabs(b) == std::numeric_limits<double>::infinity()) {
            return ((a > 0 && b > 0) || (a < 0 && b < 0)) ? std::numeric_limits<double>::infinity()
                                                          : -std::numeric_limits<double>::infinity();
        }

        if (b != 0 && std::fabs(a) > std::numeric_limits<double>::max() / std::fabs(b)) {
            return ((a > 0 && b > 0) || (a < 0 && b < 0)) ? std::numeric_limits<double>::infinity()
                                                          : -std::numeric_limits<double>::infinity();
        }

        return a * b;
    };

    auto add_check_overflow = [](double a, double b) -> double {
        if (((a > 0 && b > 0) || (a < 0 && b < 0)) &&
            std::fabs(a) > std::numeric_limits<double>::max() - std::fabs(b)) {
            return a > 0 && b > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        }

        return a + b;
    };

    log::debug() << "make_gna_pwl\n";
    log::debug() << "   in_scale  " << in_scale << "\n";
    log::debug() << "   out_scale " << out_scale << "\n";
    print_segments_header();
    gna_pwl.resize(0);
    for (size_t i = 0; i < count; i++) {
        auto s = gna_slope(m[i], in_scale, out_scale);
        int32_t xbase = (cast_check_overflow<int32_t>(mul_check_overflow(in_scale, alpha[i]), false) & XBASEMASK) |
                        s.slope_scale_index;
        int16_t ybase = cast_check_overflow<int16_t>(
            mul_check_overflow(add_check_overflow(mul_check_overflow(m[i], alpha[i]), b[i]), out_scale));
        int16_t slope = cast_check_overflow<int16_t>(mul_check_overflow(s.slope, static_cast<double>(s.slope_scale)));
        gna_pwl.push_back({xbase, ybase, slope});
        print_segment(alpha[i], add_check_overflow(mul_check_overflow(m[i], alpha[i]), b[i]), m[i]);
    }
}

static void make_gna_pwl(std::tuple<>&&,
                         const std::shared_ptr<ov::op::v0::Constant>&,
                         const std::shared_ptr<ov::op::v0::Constant>&,
                         const std::shared_ptr<ov::op::v0::Constant>&,
                         double,
                         double,
                         std::vector<gna_pwl_segment_t>&) {}

template <typename T, typename... Types>
static void make_gna_pwl(std::tuple<T, Types...>&& types,
                         const std::shared_ptr<ov::op::v0::Constant>& m,
                         const std::shared_ptr<ov::op::v0::Constant>& b,
                         const std::shared_ptr<ov::op::v0::Constant>& alpha,
                         double in_scale,
                         double out_scale,
                         std::vector<gna_pwl_segment_t>& gna_pwl) {
    if (m->get_element_type() != T::value) {
        make_gna_pwl(std::tuple<Types...>(), m, b, alpha, in_scale, out_scale, gna_pwl);
        return;
    }

    using A = typename ngraph::element_type_traits<T::value>::value_type;
    make_gna_pwl(m->get_data_ptr<A>(),
                 b->get_data_ptr<A>(),
                 alpha->get_data_ptr<A>(),
                 m->get_byte_size() / sizeof(A),
                 in_scale,
                 out_scale,
                 gna_pwl);
}

void make_gna_pwl(const std::shared_ptr<ngraph::Node>& node,
                  double in_scale,
                  double out_scale,
                  bool low_precision,
                  std::vector<gna_pwl_segment_t>& gna_pwl) {
    auto m_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
    IE_ASSERT(!!m_node);
    auto b_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(2));
    IE_ASSERT(!!b_node);
    auto alpha_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node->get_input_node_shared_ptr(3));
    IE_ASSERT(!!alpha_node);
    IE_ASSERT(m_node->get_element_type() == b_node->get_element_type() &&
              m_node->get_element_type() == alpha_node->get_element_type());
    make_gna_pwl(std::tuple<std::integral_constant<ngraph::element::Type_t, ngraph::element::f32>,
                            std::integral_constant<ngraph::element::Type_t, ngraph::element::f64>>(),
                 m_node,
                 b_node,
                 alpha_node,
                 in_scale,
                 out_scale,
                 gna_pwl);
}
