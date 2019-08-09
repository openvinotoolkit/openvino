// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//  pwl_design.cpp : simple activation function designer
//

#include "pwl.h"
#include "gna_plugin_log.hpp"
#include <vector>
#include <algorithm>
#include <limits>

#define FLOAT_TO_INT16(a) static_cast<int16_t>(((a) < 0)?((a) - 0.5):((a) + 0.5))
#define FLOAT_TO_INT32(a) static_cast<int32_t>(((a) < 0)?((a)-0.5):((a)+0.5))
#ifdef _NO_MKL_
#include <cmath>
#include <details/ie_exception.hpp>
#define SCOPY(num, in, inci, out, inco) for (int i_ = 0; i_ < *(num); i_++) *(out + i_ * *(inco)) = *(in + i_ * *(inci));
#define SSCAL(num, scale, inout, inco)  for (int i_ = 0; i_ < *(num); i_++) *(inout + i_ * *(inco)) = *(scale) * *(inout + i_ * *(inco));
#define TANH(num, in, out) for (int i_ = 0; i_ < num; i_++) *(out+i_) = tanh(*(in+i_))
#else
#include <mkl.h>
#define SCOPY(num, in, incx, out, incy) scopy(num, in, incx, out, incy)
#define SSCAL(num, scale, inout, incx) sscal(num, scale, inout, incx)
#define TANH(num, in, out) vsTanh(num, in, out)
#endif

double first_deriv_tanh(const double x) { return(1.0 - tanh(x) * tanh(x)); }

double sigmoid(const double x) { return(0.5 * (1.0 + tanh(x / 2))); }
double first_deriv_sigmoid(const double x) { return(sigmoid(x) * (1.0 - sigmoid(x))); }
double relu(const double x) { if (x < 0) { return(0.0); } else { return(x); } }
double leaky_relu(const double x) { if (x < 0.0) { return(LEAKYRELU_SLOPE*x); } else { return(x); } }
double clipping(const double x, const double lbound, const double ubound) { return((x < lbound)?lbound:((x > ubound)?ubound:x)); }

double pivot_search(std::vector<pwl_t>& result, double(*f)(const double),
                                    double(*first_deriv_f)(const double),
                                    const uint32_t N,
                                    const double alpha_0,
                                    const double alpha_N,
                                    const double threshold,
                                    const bool negative) {
    std::vector<std::vector<double>> t(N + 1);
    std::vector<std::vector<double>> alpha(N + 1);
    std::vector<std::vector<double>> epsilon(N + 1);
    std::vector<std::vector<double>> d(N + 1);
    bool same_epsilon = false;
    double Delta;
    double epsilon_final = 0.0;
    double max_epsilon = 0.0;
    double max_epsilon_prev;
    double min_epsilon;
    double sgn = (negative) ? -1.0 : 1.0;
    int j;

    if ( f == nullptr ||
        first_deriv_f == nullptr ||
        threshold < 0) {
        return epsilon_final;
    }
    // Figure 4:  Box #1
    j = 0;
    Delta = 1.0;

    for (int i = 0; i < N; i++) {
        t[i].push_back(alpha_0 + (static_cast<double>((i + 1)) / static_cast<double>((N + 1))) * (alpha_N - alpha_0));
    }

    while (true) {
        // Figure 4:  Box #2
        alpha[0].resize(j + 1);
        alpha[0][j] = alpha_0;
        for (int i = 1; i < N; i++) {
            alpha[i].resize(j + 1);
            alpha[i][j] = (f(t[i - 1][j]) - f(t[i][j]) + first_deriv_f(t[i][j]) * t[i][j] - first_deriv_f(t[i - 1][j]) * t[i - 1][j])
                / (first_deriv_f(t[i][j]) - first_deriv_f(t[i - 1][j]));
        }
        alpha[N].resize(j + 1);
        alpha[N][j] = alpha_N;

        // Figure 4:  Box #3
        for (int i = 0; i < N; i++) {
            epsilon[i].resize(j + 1);
            epsilon[i][j] = sgn * (first_deriv_f(t[i][j]) * (alpha[i][j] - t[i][j]) + f(t[i][j]) - f(alpha[i][j]));
        }
        epsilon[N].resize(j + 1);
        epsilon[N][j] = sgn * (first_deriv_f(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) + f(t[N - 1][j]) - f(alpha[N][j]));

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = fabs(epsilon[0][j]);
        min_epsilon = fabs(epsilon[0][j]);
        for (int i = 1; i < N + 1; i++) {
            if (fabs(epsilon[i][j]) > max_epsilon) max_epsilon = fabs(epsilon[i][j]);
            if (fabs(epsilon[i][j]) < min_epsilon) min_epsilon = fabs(epsilon[i][j]);
        }
        if ((j == PWL_MAX_ITERATIONS) || (max_epsilon - min_epsilon < threshold * min_epsilon)) {
            pwl_t value;
            result.resize(0);
            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (int i = 0; i < N; i++) {
                double val, val_next;
                value.t = t[i][j];
                value.alpha = alpha[i][j];
                val = sgn * first_deriv_f(value.t) * (value.alpha - value.t) + sgn * f(value.t) - epsilon_final;
                val_next = sgn * first_deriv_f(value.t) * (alpha[i + 1][j] - value.t) + sgn * f(value.t) - epsilon_final;
                value.beta = val;
                value.m = (val_next - val) / (alpha[i + 1][j] - value.alpha);
                value.b = (val - value.m * value.alpha);
                result.push_back(value);
            }
            value.t = value.m = value.b = 0.0;
            value.alpha = alpha[N][j];
            value.beta = sgn * first_deriv_f(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) + sgn * f(t[N - 1][j]) - epsilon_final;
            result.push_back(value);
            if (j == PWL_MAX_ITERATIONS) {
                std::cerr << "Error:  failed to converge in pivot_search!" << std::endl;
            }
            return(epsilon_final);
        }

        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j = j - 1;
                Delta = Delta / 2;
            } else if (max_epsilon == max_epsilon_prev) {
                if (!same_epsilon) {
                    same_epsilon = true;
                } else {
                    j = j - 1;
                    Delta = Delta / 2;
                    same_epsilon = false;
                }
            }
        }

        // Figure 4:  Box #4
        for (int i = 0; i < N; i++) {
            d[i].resize(j + 1);
            d[i][j] = Delta * (epsilon[i + 1][j] - epsilon[i][j]) /
                ((epsilon[i + 1][j] / (alpha[i + 1][j] - t[i][j])) + (epsilon[i][j] / (t[i][j] - alpha[i][j])));
        }

        // Figure 4:  Box #5
        for (int i = 0; i < N; i++) {
            t[i].resize(j + 2);
            t[i][j + 1] = t[i][j] + d[i][j];
        }
        t[N].resize(j + 2);

        j = j + 1;
    }
}

double calculate_error_pct(const DnnActivationType fun,
                            const double l_bound,
                            const double u_bound,
                            const double offset,
                            const int samples) {
    double delta = (u_bound - l_bound) / (samples + 1);
    double min_val = 0.0;
    double max_val = 0.0;

    if ( delta < 0 ) {
        return 0.0;
    }

    switch (fun) {
        case kActSigmoid:  min_val = max_val = sigmoid(l_bound); break;
        case kActTanh:     min_val = max_val = tanh(l_bound); break;
    }

    for (int i = 0; i < samples; i++) {
        double arg = l_bound + i * delta;
        double val = 0.0;
        switch (fun) {
            case kActSigmoid:  val = sigmoid(arg); break;
            case kActTanh:     val = tanh(arg); break;
        }
        if (val > max_val) max_val = val;
        if (val < min_val) min_val = val;
    }

    return(100.0 * fabs(offset) / (max_val - min_val));
}

bool split_search(const DnnActivationType fun,
                    const double l_bound,
                    const double u_bound) {
    bool is_split = false;
    if (l_bound > u_bound) {
        return is_split;
    }

    switch (fun) {
        case kActSigmoid:
        case kActTanh:
            if ((l_bound < 0.0) && (u_bound > 0.0)) {
                is_split = true;
            }
            break;
        default:
            is_split = false;
    }
    return(is_split);
}

inline std::vector<pwl_t> negative_pwl(const std::vector<pwl_t>& pwl) {
    std::vector<pwl_t> new_pwl;
    new_pwl = pwl;
    for (uint32_t i = 0; i < pwl.size(); i++) {
        new_pwl[i].m = -pwl[i].m;
        new_pwl[i].b = -pwl[i].b;
        new_pwl[i].beta = -pwl[i].beta;
    }

    return(new_pwl);
}

std::vector<pwl_t> pwl_search(const DnnActivationType fun,
                                const double l_bound,
                                const double u_bound,
                                const double threshold,
                                const double allowed_err_pct,
                                const int samples,
                                double& err_pct) {
    std::vector<pwl_t> pwl;
    double err = 0.0;
    int n_segments = 1;

    if (l_bound > u_bound ||
        threshold < 0) {
        return pwl;
    }

    if (split_search(fun, l_bound, u_bound)) {
        std::vector<pwl_t> pwl2;
        double err_pct1 = 0.0, err_pct2 = 0.0;

        pwl = pwl_search(fun, l_bound, 0.0, threshold, allowed_err_pct, samples, err_pct1);
        pwl = negative_pwl(pwl);
        pwl2 = pwl_search(fun, 0.0, u_bound, threshold, allowed_err_pct, samples, err_pct2);

        // merge
        pwl.pop_back();  // remove final alpha and beta from first half
        pwl.insert(pwl.end(), pwl2.begin(), pwl2.end());  // concatenate the two halves
        err_pct = (err_pct1 + err_pct2) / 2;  // this is not quite correct but should give an indication

    } else {
        if (fun == kActIdentity) {
            pwl.resize(2);
            pwl[0].alpha = pwl[0].t = pwl[0].beta = -std::numeric_limits<float>::infinity();
            pwl[0].m = 1.0;
            pwl[0].b = 0.0;
            pwl[1].alpha = std::numeric_limits<float>::infinity();
            pwl[1].beta = std::numeric_limits<float>::infinity();

        } else if (fun == kActKaldiLstmClipping) {
            pwl.resize(4);
            pwl[0].alpha = pwl[0].t = pwl[0].beta = -std::numeric_limits<float>::infinity();
            pwl[0].m = 0.0;
            pwl[0].b = pwl[0].beta = KALDI_LSTM_CLIP_LOWER;
            pwl[1].alpha = pwl[0].t = pwl[1].beta = KALDI_LSTM_CLIP_LOWER;
            pwl[1].m = 1.0;
            pwl[1].b = 0.0;
            pwl[2].alpha = pwl[0].t = pwl[1].beta = KALDI_LSTM_CLIP_UPPER;
            pwl[2].m = 0.0;
            pwl[2].b = KALDI_LSTM_CLIP_UPPER;
            pwl[3].alpha = pwl[3].beta = std::numeric_limits<float>::infinity();

        } else {
            bool negative = false;

            switch (fun) {
                case kActSigmoid:
                    if (u_bound == 0) negative = true;  // make left half convex
                    err = pivot_search(pwl, sigmoid, first_deriv_sigmoid, n_segments, l_bound, u_bound, threshold, negative);
                    break;
                case kActTanh:
                    if (u_bound == 0) negative = true;  // make left half convex
                    err = pivot_search(pwl, tanh, first_deriv_tanh, n_segments, l_bound, u_bound, threshold, negative);
                    break;
            }
            err_pct = calculate_error_pct(fun, l_bound, u_bound, err, samples);

            while ((n_segments < PWL_MAX_ITERATIONS) && (allowed_err_pct < err_pct)) {
                n_segments += 1;
                switch (fun) {
                    case kActSigmoid:
                        err = pivot_search(pwl, sigmoid, first_deriv_sigmoid, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                    case kActTanh:
                        err = pivot_search(pwl, tanh, first_deriv_tanh, n_segments, l_bound, u_bound, threshold, negative);
                        break;
                }
                err_pct = calculate_error_pct(fun, l_bound, u_bound, err, samples);
            }

            if (n_segments >= PWL_MAX_ITERATIONS) {
                std::cerr << "Error:  failed to converge in pwl_search!" << std::endl;
            }
        }
    }
    return(pwl);
}

pwl_gna_slope_scale_t gna_slope(const double slope,
                                const double in_scale,
                                const double out_scale) {
    pwl_gna_slope_scale_t s;
    s.slope = slope* out_scale / in_scale;

    for (s.slope_scale_index = 3; s.slope_scale_index > 0; --s.slope_scale_index) {
        s.slope_scale = static_cast<uint64_t>(1) << (8 * (1 + s.slope_scale_index));
        if (((s.slope * s.slope_scale) <= std::numeric_limits<int16_t>::max()) &&
                    ((s.slope * s.slope_scale) >= std::numeric_limits<int16_t>::min()))
            break;
    }
    s.slope_scale = static_cast<uint64_t>(1) << (8 * (1 + s.slope_scale_index));

    return(s);
}

void make_gna_pwl(const DnnActivation  fun,
                    const std::vector<pwl_t>& pwl,
                    const double l_bound,
                    const double u_bound,
                    const double in_scale,
                    const double out_scale,
                    std::vector<intel_pwl_segment_t> &gna_pwl) {
    pwl_gna_slope_scale_t s;
    uint32_t pwl_size = static_cast<int32_t>(pwl.size());
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
            } else {
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
            gnalog() << gna_pwl[1].xBase / in_scale
                    << " " << gna_pwl[1].yBase / out_scale
                    << " " << 1.0
                    << "\n";
            if (INT32_MAX > x_upper) {  // need a right segment
                gna_pwl.push_back({
                    static_cast<int32_t>(x_upper & XBASEMASK),  // zero out the 2 lsb
                    y_upper,
                    0 });

                gnalog() << gna_pwl[n_segments].xBase / in_scale
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

void PwlDesignOpt16(const DnnActivation activation_type,
                    std::vector<intel_pwl_segment_t> &ptr_segment,
                    const float scale_in,
                    const float scale_out) {
    std::vector<pwl_t> pwl;
    double err_pct = 0.0;
    switch (activation_type) {
        case kActSigmoid:
            pwl = pwl_search(kActSigmoid, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, PWL_DESIGN_THRESHOLD, PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            make_gna_pwl(activation_type, pwl, -SIGMOID_DOMAIN, SIGMOID_DOMAIN, scale_in, scale_out, ptr_segment);
            break;
        case kActTanh:
            pwl = pwl_search(kActTanh, -TANH_DOMAIN, TANH_DOMAIN, PWL_DESIGN_THRESHOLD, PWL_MAX_ERR_PERCENT, PWL_DESIGN_SAMPLES, err_pct);
            make_gna_pwl(activation_type, pwl, -TANH_DOMAIN, TANH_DOMAIN, scale_in, scale_out, ptr_segment);
            break;
        case kActRelu:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment);
            break;
        case kActLeakyRelu:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment);
            break;
        case kActIdentity:
            make_gna_pwl(activation_type, pwl, -1.0, 1.0, scale_in, scale_out, ptr_segment);
            break;
        case kActKaldiLstmClipping:
            make_gna_pwl(activation_type, pwl, KALDI_LSTM_CLIP_LOWER, KALDI_LSTM_CLIP_UPPER, scale_in, scale_out, ptr_segment);
            break;
        default:
            break;
    }
}

void PwlDesign16(const DnnActivation activation_type,
                 intel_pwl_segment_t *ptr_segment,
                 const uint32_t num_segments,
                 const float scale_in,
                 const float scale_out) {
    switch (activation_type) {
        case kActSigmoid:
           {
                gnalog() <<  "=========================== Sigmoid Segments===========================\n";
                uint32_t num_segment_size = 0;
                int32_t offset = 0;
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                num_segment_size = static_cast<int32_t>(SIGMOID_DOMAIN * scale_in / ((num_segments-2) / 2) + 0.5);
                offset = -static_cast<int32_t>(num_segment_size * (num_segments-2) / 2);
                for (uint32_t i = 1; i < num_segments; i++) {
                    ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
                    offset += num_segment_size;
                }
                for (uint32_t i = 0; i < num_segments; i++) {
                    int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
                    int32_t xbasenext = (i < num_segments-1) ? static_cast<int32_t>(ptr_segment[i+1].xBase & XBASEMASK) : INT32_MAX;
                    float floatarg = static_cast<float>(xbase / (2 * scale_in));
                    float floatargnext = static_cast<float>(xbasenext / (2 * scale_in));
                    float floatval, floatvalnext, slope;
                    TANH(1, &floatarg, &floatval);
                    floatval = 0.5f * (1.0f + floatval);
                    TANH(1, &floatargnext, &floatvalnext);
                    floatvalnext = 0.5f * (1.0f + floatvalnext);
                    slope = scale_out*(floatvalnext - floatval) / static_cast<float>(xbasenext - xbase);
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
                        ptr_segment[i].slope = FLOAT_TO_INT16(slope * slope_scale);

                        ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
                    }
                    ptr_segment[i].yBase = FLOAT_TO_INT16(floatval * scale_out);
                    gnalog() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK))/scale_out)
                             << " "
                             << (static_cast<float>((ptr_segment[i].yBase))/scale_out)
                             << " "
                             << (slope/scale_out)
                             << "\n";
                }
            }
            break;
        case kActTanh:
            {
                gnalog() <<  "=========================== Tanh Segments===========================\n";
                uint32_t num_segment_size = 0;
                int32_t offset = 0;
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                num_segment_size = static_cast<int32_t>(TANH_DOMAIN * scale_in / ((num_segments-2) / 2) + 0.5);
                offset = -static_cast<int32_t>(num_segment_size * (num_segments-2) / 2);
                for (uint32_t i = 1; i < num_segments; i++) {
                    ptr_segment[i].xBase = static_cast<int32_t>(offset & XBASEMASK);  // zero out the 2 lsb
                    offset += num_segment_size;
                }
                for (uint32_t i = 0; i < num_segments; i++) {
                    int32_t xbase = static_cast<int32_t>(ptr_segment[i].xBase & XBASEMASK);
                    int32_t xbasenext = (i < num_segments-1) ?
                                                    static_cast<int32_t>(ptr_segment[i+1].xBase & XBASEMASK) :
                                                    INT32_MAX;
                    float floatarg = static_cast<float>(xbase / scale_in);
                    float floatargnext = static_cast<float>(xbasenext / scale_in);
                    float floatval, floatvalnext, slope;
                    TANH(1, &floatarg, &floatval);
                    TANH(1, &floatargnext, &floatvalnext);
                    slope = scale_out * (floatvalnext - floatval) /
                                                static_cast<float>(xbasenext - xbase);
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
                        ptr_segment[i].slope = FLOAT_TO_INT16(slope * slope_scale);
                        ptr_segment[i].xBase = ptr_segment[i].xBase | slope_scale_index;
                    }
                    ptr_segment[i].yBase = FLOAT_TO_INT16(floatval * scale_out);
                    gnalog() << (static_cast<int32_t>((ptr_segment[i].xBase & XBASEMASK))/scale_out)
                             << " "
                             << (static_cast<float>((ptr_segment[i].yBase))/scale_out)
                             << " "
                             << (slope/scale_out)
                             << "\n";
                }
            }
            break;
        case kActRelu:
            std::cerr << "Rectilinear activation function design not yet implemented!" << std::endl;
            throw -1;
        case kActIdentity:
        case kActKaldiLstmClipping:  // clipping of IDENTITY is more aggressive than Kaldi
            {
                float slope = 0.0;
                int64_t x_lower_limit = static_cast<int64_t>((INT16_MIN / scale_out) * scale_in - 0.5);
                int64_t x_upper_limit = static_cast<int64_t>((INT16_MAX / scale_out) * scale_in + 0.5);
                int16_t y_lower_limit = INT16_MIN;
                int16_t y_upper_limit = INT16_MAX;
                if (activation_type == kActKaldiLstmClipping)
                    gnalog() << "=========================== Clipping Segments ===========================\n";
                else
                    gnalog() << "=========================== Identity Segments ===========================\n";
                if (x_lower_limit < INT32_MIN) {
                    std::cerr << "Warning:  saturation in PwlDesign16! " << x_lower_limit  << " < INT32_MIN"<< std::endl;
                    x_lower_limit = INT32_MIN;
                    y_lower_limit = static_cast<int16_t>((scale_out / scale_in)*static_cast<float>(INT32_MIN) - 0.5);
                }
                if (x_upper_limit > INT32_MAX) {
                    std::cerr << "Warning:  saturation in PwlDesign16! " << x_upper_limit  << " > INT32_MAX"<< std::endl;
                    x_upper_limit = INT32_MAX;
                    y_upper_limit = static_cast<int16_t>((scale_out / scale_in)*static_cast<float>(INT32_MAX) + 0.5);
                }
                slope =
                    static_cast<float>(static_cast<uint64_t>(y_upper_limit) - static_cast<uint64_t>(y_lower_limit)) /
                                               static_cast<float>(static_cast<uint64_t>(x_upper_limit) - static_cast<uint64_t>(x_lower_limit));
                ptr_segment[0].xBase = static_cast<int32_t>(INT32_MIN & XBASEMASK);  // zero out the 2 lsb
                ptr_segment[0].yBase = y_lower_limit;
                ptr_segment[0].slope = 0;

                gnalog() << ptr_segment[0].xBase / scale_in
                    << " " << ptr_segment[0].yBase / scale_out
                    << " " << 0
                    << "\n";

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
                    ptr_segment[1].slope = FLOAT_TO_INT16(slope * slope_scale);
                    ptr_segment[1].xBase = ptr_segment[1].xBase | slope_scale_index;
                }
                ptr_segment[2].xBase = static_cast<int32_t>(x_upper_limit & XBASEMASK);
                ptr_segment[2].yBase = y_upper_limit;
                ptr_segment[2].slope = 0;
            }
            break;
        default:
            fprintf(stderr, "Activation function design for %s not yet implemented!\n", intel_dnn_activation_name[activation_type]);
            throw -1;
    }
}
