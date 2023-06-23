// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//  pwl_design.cpp : simple activation function designer
//

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "frontend/quantization.hpp"

#ifdef _NO_MKL_
#    include <cmath>

#    include "backend/make_pwl.hpp"

#    define SCOPY(num, in, inci, out, inco) \
        for (int i_ = 0; i_ < *(num); i_++) \
            *(out + i_ * *(inco)) = *(in + i_ * *(inci));
#    define SSCAL(num, scale, inout, inco)  \
        for (int i_ = 0; i_ < *(num); i_++) \
            *(inout + i_ * *(inco)) = *(scale) * *(inout + i_ * *(inco));
#    define TANH(num, in, out)           \
        for (int i_ = 0; i_ < num; i_++) \
        *(out + i_) = tanh(*(in + i_))
#else
#    include <mkl.h>
#    define SCOPY(num, in, incx, out, incy) scopy(num, in, incx, out, incy)
#    define SSCAL(num, scale, inout, incx)  sscal(num, scale, inout, incx)
#    define TANH(num, in, out)              vsTanh(num, in, out)
#endif

#include "common/numerical_utils.hpp"
#include "gna_slope_scale.hpp"
#include "log/debug.hpp"
#include "log/log.hpp"
#include "ops/reference/pwl.hpp"
#include "pwl.h"

using namespace ov::intel_gna;
using namespace ov::intel_gna::common;

double relu(const double x) {
    if (x < 0) {
        return (0.0);
    } else {
        return (x);
    }
}
double leaky_relu(const double x) {
    if (x < 0.0) {
        return (LEAKYRELU_SLOPE * x);
    } else {
        return (x);
    }
}
double clipping(const double x, const double lbound, const double ubound) {
    return ((x < lbound) ? lbound : ((x > ubound) ? ubound : x));
}

inline double power(const double x, const std::tuple<double, double, double>& args) {
    return (pow(std::get<2>(args) + std::get<1>(args) * x, std::get<0>(args)));
}

void PwlDesignOpt(const DnnActivation& activation_type,
                  const float scale_in,
                  const float scale_out,
                  const bool low_precision,
                  const std::shared_ptr<ngraph::Node>& node,
                  const bool is_fused_with_conv2d,
                  std::vector<gna_pwl_segment_t>& ptr_segment) {
    std::vector<pwl_t> pwl;
    switch (activation_type) {
    case kActPwl: {
        make_gna_pwl(node, scale_in, scale_out, low_precision, ptr_segment);
        break;
    }
    case kActRelu:
        make_gna_pwl(activation_type,
                     pwl,
                     -1.0,
                     1.0,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    case kActLeakyRelu:
        make_gna_pwl(activation_type,
                     pwl,
                     -1.0,
                     1.0,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    case kActIdentity:
    case kActFakeQuantize:
        make_gna_pwl(activation_type,
                     pwl,
                     -1.0,
                     1.0,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    case kActKaldiLstmClipping:
        make_gna_pwl(activation_type,
                     pwl,
                     activation_type.args.clamp.low,
                     activation_type.args.clamp.high,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    case kActSign:
        make_gna_pwl(activation_type,
                     pwl,
                     -1.0,
                     1.0,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    case kActAbs:
        make_gna_pwl(activation_type,
                     pwl,
                     -1.0,
                     1.0,
                     scale_in,
                     scale_out,
                     low_precision,
                     is_fused_with_conv2d,
                     ptr_segment);
        break;
    default:
        THROW_GNA_EXCEPTION << "Unknown piecewise linear function type: " << activation_type.type;
    }
}

void PwlDesign(const DnnActivation& activation_type,
               gna_pwl_segment_t* ptr_segment,
               const uint32_t num_segments,
               const float scale_in,
               const float scale_out,
               const bool low_precision) {
    switch (activation_type) {
    case kActSigmoid: {
        log::debug() << "=========================== Sigmoid Segments===========================\n";
        uint32_t num_segment_size = 0;
        int32_t offset = 0;
        ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
        num_segment_size = static_cast<int32_t>(SIGMOID_DOMAIN * scale_in / ((num_segments - 2) / 2) + 0.5);
        offset = -static_cast<int32_t>(num_segment_size * (num_segments - 2) / 2);
        for (uint32_t i = 1; i < num_segments; i++) {
            ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
            offset += num_segment_size;
        }
        for (uint32_t i = 0; i < num_segments; i++) {
            int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
            int32_t xbasenext =
                (i < num_segments - 1) ? static_cast<int32_t>(ptr_segment[i + 1].xBase & XBASEMASK) : INT32_MAX;
            float floatarg = static_cast<float>(xbase / (2 * scale_in));
            float floatargnext = static_cast<float>(xbasenext / (2 * scale_in));
            float floatval, floatvalnext, slope;
            TANH(1, &floatarg, &floatval);
            floatval = 0.5f * (1.0f + floatval);
            TANH(1, &floatargnext, &floatvalnext);
            floatvalnext = 0.5f * (1.0f + floatvalnext);
            slope = scale_out * (floatvalnext - floatval) / static_cast<float>(xbasenext - xbase);
            {
                // find best scale factor
                uint64_t slope_scale;
                uint32_t slope_scale_index;
                for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                    slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                    if (((slope * slope_scale) <= 32767.0) && ((slope * slope_scale) >= -32768.0))
                        break;
                }
                slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                ptr_segment[i].slope = FloatToInt16(slope * slope_scale);

                ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
            }
            ptr_segment[i].yBase = FloatToInt16(floatval * scale_out);
            log::debug() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK)) / scale_out) << " "
                         << (static_cast<float>((ptr_segment[i].yBase)) / scale_out) << " " << (slope / scale_out)
                         << "\n";
        }
    } break;
    case kActTanh: {
        log::debug() << "=========================== Tanh Segments===========================\n";
        uint32_t num_segment_size = 0;
        int32_t offset = 0;
        ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
        num_segment_size = static_cast<int32_t>(TANH_DOMAIN * scale_in / ((num_segments - 2) / 2) + 0.5);
        offset = -static_cast<int32_t>(num_segment_size * (num_segments - 2) / 2);
        for (uint32_t i = 1; i < num_segments; i++) {
            ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
            offset += num_segment_size;
        }
        for (uint32_t i = 0; i < num_segments; i++) {
            int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
            int32_t xbasenext =
                (i < num_segments - 1) ? static_cast<int32_t>(ptr_segment[i + 1].xBase & XBASEMASK) : INT32_MAX;
            float floatarg = static_cast<float>(xbase / scale_in);
            float floatargnext = static_cast<float>(xbasenext / scale_in);
            float floatval, floatvalnext, slope;
            TANH(1, &floatarg, &floatval);
            TANH(1, &floatargnext, &floatvalnext);
            slope = scale_out * (floatvalnext - floatval) / static_cast<float>(xbasenext - xbase);
            {
                // find best scale factor
                uint64_t slope_scale;
                uint32_t slope_scale_index;
                for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                    slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                    if (((slope * slope_scale) <= 32767.0) && ((slope * slope_scale) >= -32768.0))
                        break;
                }
                slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                ptr_segment[i].slope = FloatToInt16(slope * slope_scale);
                ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
            }
            ptr_segment[i].yBase = FloatToInt16(floatval * scale_out);
            log::debug() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK)) / scale_out) << " "
                         << (static_cast<float>((ptr_segment[i].yBase)) / scale_out) << " " << (slope / scale_out)
                         << "\n";
        }
    } break;
    case kActSoftSign: {
        auto softsign = [](const double x) {
            return (x / (1.0 + fabs(x)));
        };
        log::debug() << "=========================== SoftSign Segments===========================\n";
        uint32_t num_segment_size = 0;
        int32_t offset = 0;
        ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
        num_segment_size = static_cast<int32_t>(SOFTSIGN_DOMAIN * scale_in / ((num_segments - 2) / 2) + 0.5);
        offset = -static_cast<int32_t>(num_segment_size * (num_segments - 2) / 2);
        for (uint32_t i = 1; i < num_segments; i++) {
            ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
            offset += num_segment_size;
        }
        for (uint32_t i = 0; i < num_segments; i++) {
            int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
            int32_t xbasenext =
                (i < num_segments - 1) ? static_cast<int32_t>(ptr_segment[i + 1].xBase & XBASEMASK) : INT32_MAX;
            float floatarg = static_cast<float>(xbase / (2 * scale_in));
            float floatargnext = static_cast<float>(xbasenext / (2 * scale_in));
            float floatval, floatvalnext, slope;
            floatval = static_cast<float>(softsign(floatarg));
            floatvalnext = static_cast<float>(softsign(floatargnext));
            slope = scale_out * (floatvalnext - floatval) / static_cast<float>(xbasenext - xbase);
            {
                // find best scale factor
                uint64_t slope_scale;
                uint32_t slope_scale_index;
                for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                    slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                    if (((slope * slope_scale) <= 32767.0) && ((slope * slope_scale) >= -32768.0))
                        break;
                }
                slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                ptr_segment[i].slope = FloatToInt16(slope * slope_scale);
                ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
            }
            ptr_segment[i].yBase = FloatToInt16(floatval * scale_out);
            log::debug() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK)) / scale_out) << " "
                         << (static_cast<float>((ptr_segment[i].yBase)) / scale_out) << " " << (slope / scale_out)
                         << "\n";
        }
    } break;
    case kActRelu:
        THROW_GNA_EXCEPTION << "Rectilinear activation function design not yet implemented!";
    case kActIdentity:
    case kActKaldiLstmClipping:  // clipping of IDENTITY is more aggressive than Kaldi
    {
        float slope = 0.0;
        int64_t x_lower_limit = static_cast<int64_t>((INT16_MIN / scale_out) * scale_in - 0.5);
        int64_t x_upper_limit = static_cast<int64_t>((INT16_MAX / scale_out) * scale_in + 0.5);
        int16_t y_lower_limit = INT16_MIN;
        int16_t y_upper_limit = INT16_MAX;
        if (activation_type == kActKaldiLstmClipping)
            log::debug() << "=========================== Clipping Segments ===========================\n";
        else
            log::debug() << "=========================== Identity Segments ===========================\n";
        if (x_lower_limit < INT32_MIN) {
            log::warning() << "Saturation in PwlDesign! " << x_lower_limit << " < INT32_MIN" << std::endl;
            x_lower_limit = INT32_MIN;
            y_lower_limit = static_cast<int16_t>((scale_out / scale_in) * static_cast<float>(INT32_MIN) - 0.5);
        }
        if (x_upper_limit > INT32_MAX) {
            log::warning() << "Saturation in PwlDesign! " << x_upper_limit << " > INT32_MAX" << std::endl;
            x_upper_limit = INT32_MAX;
            y_upper_limit = static_cast<int16_t>((scale_out / scale_in) * static_cast<float>(INT32_MAX) + 0.5);
        }
        slope = static_cast<float>(static_cast<uint64_t>(y_upper_limit) - static_cast<uint64_t>(y_lower_limit)) /
                static_cast<float>(static_cast<uint64_t>(x_upper_limit) - static_cast<uint64_t>(x_lower_limit));
        ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
        ptr_segment[0].yBase = y_lower_limit;
        ptr_segment[0].slope = 0;

        log::debug() << ptr_segment[0].xBase / scale_in << " " << ptr_segment[0].yBase / scale_out << " " << 0 << "\n";

        ptr_segment[1].xBase = static_cast<int32_t>(x_lower_limit & XBASEMASK);
        ptr_segment[1].yBase = y_lower_limit;
        {
            // find best scale factor
            uint64_t slope_scale = 0;
            uint32_t slope_scale_index = 0;
            for (slope_scale_index = 3; slope_scale_index > 0; slope_scale_index--) {
                slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
                if (((slope * slope_scale) <= std::numeric_limits<int16_t>::max()) &&
                    ((slope * slope_scale) >= std::numeric_limits<int16_t>::min()))
                    break;
            }
            slope_scale = static_cast<uint64_t>(1) << (8 * (1 + slope_scale_index));
            ptr_segment[1].slope = FloatToInt16(slope * slope_scale);
            ptr_segment[1].xBase = ptr_segment[1].xBase | slope_scale_index;
        }
        ptr_segment[2].xBase = static_cast<int32_t>(x_upper_limit & XBASEMASK);
        ptr_segment[2].yBase = y_upper_limit;
        ptr_segment[2].slope = 0;
    } break;
    case kActPow: {
        log::debug() << "=========================== Pow Segments===========================\n";
        uint32_t num_segment_size = 0;

        auto args = std::tuple<double, double, double>{activation_type.args.pow.exponent,
                                                       activation_type.args.pow.scale,
                                                       activation_type.args.pow.offset};

        auto input_min_value = static_cast<double>(std::numeric_limits<int32_t>::min());
        auto input_max_value = static_cast<double>(std::numeric_limits<int32_t>::max());
        double x_min = AreFpEq(fmod(activation_type.args.pow.exponent, 1.0), 0.0) ? input_min_value / scale_in : 0.0;
        x_min = std::max(x_min, -POW_DOMAIN);

        double x_max = input_max_value / scale_in;
        x_max = std::min(x_max, POW_DOMAIN);

        double pow_domain = x_max - x_min;
        ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
        num_segment_size = static_cast<int32_t>(pow_domain * scale_in / (num_segments - 2) + 0.5);
        int32_t x_min_scaled = static_cast<int32_t>(x_min * scale_in + 0.5);
        int32_t offset = x_min_scaled;
        for (uint32_t i = 1; i < num_segments; i++) {
            ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
            offset += num_segment_size;
        }
        for (uint32_t i = 0; i < num_segments; i++) {
            int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
            int32_t xbasenext =
                (i < num_segments - 1) ? static_cast<int32_t>(ptr_segment[i + 1].xBase & XBASEMASK) : INT32_MAX;

            double arg = xbase / scale_in;
            arg = arg < x_min ? x_min : arg;

            double argnext = xbasenext / scale_in;
            argnext = argnext < x_min ? x_min : argnext;

            double val = power(arg, args);
            double valnext = power(argnext, args);

            double slope = (valnext - val) / (static_cast<double>(xbasenext - xbase) / scale_in);
            auto s = gna_slope(slope, scale_in, scale_out);

            ptr_segment[i].slope = DoubleToInt16(s.slope * s.slope_scale);
            ptr_segment[i].xBase = ptr_segment[i].xBase | s.slope_scale_index;

            ptr_segment[i].yBase = DoubleToInt16(val * scale_out);
            log::debug() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK)) / scale_out) << " "
                         << (static_cast<float>((ptr_segment[i].yBase)) / scale_out) << " " << (s.slope / scale_out)
                         << "\n";
        }
    } break;
    default:
        fprintf(stderr,
                "Activation function design for %s not yet implemented!\n",
                intel_dnn_activation_name[activation_type]);
        throw - 1;
    }
}

void PwlApply32(intel_dnn_component_t* component, uint32_t num_subset_size) {
    if (component->orientation_in == kDnnInterleavedOrientation) {  // subsets only supported in interleaved orientation
        PwlApply32(component, 0, num_subset_size - 1, 0, component->num_columns_in - 1);
    } else {
        PwlApply32(component, 0, component->num_rows_in - 1, 0, component->num_columns_in - 1);
    }
}

void PwlApply32(intel_dnn_component_t* component,
                uint32_t num_row_start,
                uint32_t num_row_end,
                uint32_t num_col_start,
                uint32_t num_col_end) {
    intel_piecewiselinear_t* transform = reinterpret_cast<intel_piecewiselinear_t*>(&component->op.pwl);
    float* ptr_in = reinterpret_cast<float*>(component->ptr_inputs);
    float* ptr_out = reinterpret_cast<float*>(component->ptr_outputs);
    uint32_t num_columns = component->num_columns_in;
    switch (transform->func_id.type) {
    case kActSigmoid:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = 0.5f * (1.0f + tanh(0.5f * ptr_in[i * num_columns + j]));
            }
        }
        break;
    case kActTanh:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = tanh(ptr_in[i * num_columns + j]);
            }
        }
        break;
    case kActSoftSign:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] =
                    static_cast<float>(ptr_in[i * num_columns + j] / (1.0 + fabs(ptr_in[i * num_columns + j])));
            }
        }
        break;
    case kActRelu:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] =
                    (ptr_in[i * num_columns + j] < 0.0f)
                        ? ptr_in[i * num_columns + j] * transform->func_id.args.lrelu.negative_slope
                        : ptr_in[i * num_columns + j];
            }
        }
        break;
    case kActIdentity:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = ptr_in[i * num_columns + j];
            }
        }
        break;
    case kActKaldiLstmClipping: {
        float upper_limit = component->op.pwl.func_id.args.clamp.high;
        float lower_limit = component->op.pwl.func_id.args.clamp.low;
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                float val = ptr_in[i * num_columns + j];
                if (val > upper_limit) {
                    ptr_out[i * num_columns + j] = upper_limit;
                } else if (val < lower_limit) {
                    ptr_out[i * num_columns + j] = lower_limit;
                } else {
                    ptr_out[i * num_columns + j] = val;
                }
            }
        }
        break;
    }
    case kActExp:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = exp(ptr_in[i * num_columns + j]);
            }
        }
        break;
    case kActLog:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = std::log(ptr_in[i * num_columns + j]);
            }
        }
        break;
    case kActAbs:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = fabs(ptr_in[i * num_columns + j]);
            }
        }
        break;
    case kActSign:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] =
                    (ptr_in[i * num_columns + j] == 0.f) ? 0.0f : ((ptr_in[i * num_columns + j] > 0) ? 1.0f : -1.0f);
            }
        }
        break;
    case kActNegLog:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = static_cast<float>(-1.0 * std::log(ptr_in[i * num_columns + j]));
            }
        }
        break;
    case kActNegHalfLog:
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] = static_cast<float>(-0.5 * std::log(ptr_in[i * num_columns + j]));
            }
        }
        break;
    case kActPow: {
        float exponent = transform->func_id.args.pow.exponent;
        float scale = transform->func_id.args.pow.scale;
        float offset = transform->func_id.args.pow.offset;
        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                ptr_out[i * num_columns + j] =
                    static_cast<float>(pow(offset + scale * ptr_in[i * num_columns + j], exponent));
            }
        }
    } break;
    case kActFakeQuantize: {
        auto levels = static_cast<uint32_t>(transform->func_id.fqParams.levels);

        for (uint32_t i = num_row_start; i <= num_row_end; i++) {
            auto inputChannel = transform->func_id.fqParams.inputPerChannel ? i : 0;
            auto outputChannel = transform->func_id.fqParams.outputPerChannel ? i : 0;

            float input_low = transform->func_id.fqParams.input_low[inputChannel];
            float input_high = transform->func_id.fqParams.input_high[inputChannel];
            float output_low = transform->func_id.fqParams.output_low[outputChannel];
            float output_high = transform->func_id.fqParams.output_high[outputChannel];

            for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                auto offset = i * num_columns + j;

                ptr_out[offset] = ov::intel_gna::frontend::ApplyFQ(ptr_in[offset],
                                                                   input_low,
                                                                   input_high,
                                                                   output_low,
                                                                   output_high,
                                                                   levels);
            }
        }
        break;
    }
    case kActCustom:
    default:
        THROW_GNA_EXCEPTION << component->original_layer_name
                            << ", Unknown piecewise linear function type: " << transform->func_id.type;
    }
}
