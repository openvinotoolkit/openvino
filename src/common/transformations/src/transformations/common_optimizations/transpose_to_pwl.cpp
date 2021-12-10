// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_to_pwl.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include <numeric>
#include <iostream>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

struct Pwl {
    double m;
    double b;
    double alpha;
};

static constexpr double EXP_BREAK = 0.045;
static constexpr double POW_BREAK = 0;
static constexpr int PWL_MAX_ITERATIONS = 2000;

namespace ngraph {
namespace pass {

NGRAPH_RTTI_DEFINITION(TransposeToPwl, "TransposeToPwl", 0);

template<typename T>
double get_break_bound() {
    if (std::is_same<T, opset1::Exp>::value) {
        return EXP_BREAK;
    }

    if (std::is_same<T, opset1::Power>::value) {
        return POW_BREAK;
    }

    return 0;
}

template<typename T>
bool split_search(double lower_bound, double upper_bound) {
    if (lower_bound > upper_bound) {
        return false;
    }
    double break_bound = get_break_bound<T>();
    if (std::is_same<T, opset1::Sigmoid>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        //std::is_same<T, opset1::SoftSign>::value ||
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Power>::value) {
        return lower_bound < break_bound && upper_bound > break_bound;
    }
    return false;
}

template<typename T>
double pivot_search(
    std::vector<Pwl>& pwl,
    uint32_t N,
    double alpha_0,
    double alpha_N,
    double threshold,
    bool negative) {
    double epsilon_final = 0.0;
    std::vector<std::vector<double>> t(N + 1);
    std::vector<std::vector<double>> alpha(N + 1);
    std::vector<std::vector<double>> epsilon(N + 1);
    std::vector<std::vector<double>> d(N + 1);
    bool same_epsilon = false;
    double max_epsilon = 0;
    double max_epsilon_prev = 0;
    double min_epsilon = 0;
    double sgn = negative ? -1.0 : 1.0;
    if (threshold < 0) {
        return epsilon_final;
    }

    // Figure 4:  Box #1
    int j = 0;
    double delta = 1.0;

    for (uint32_t i = 0; i < N; i++) {
        t[i].push_back(alpha_0 + (static_cast<double>((i + 1)) / static_cast<double>((N + 1))) * (alpha_N - alpha_0));
    }

    while (true) {
        // Figure 4:  Box #2
        alpha[0].resize(j + 1);
        alpha[0][j] = alpha_0;
        for (uint32_t i = 1; i < N; i++) {
            alpha[i].resize(j + 1);
            alpha[i][j] =
                (details::function<T>(t[i - 1][j]) - details::function<T>(t[i][j]) +
                    details::first_derivative<T>(t[i][j]) * t[i][j] - details::first_derivative<T>(t[i - 1][j]) * t[i - 1][j])
                / (details::first_derivative<T>(t[i][j]) - details::first_derivative<T>(t[i - 1][j]));
        }

        alpha[N].resize(j + 1);
        alpha[N][j] = alpha_N;

        // Figure 4:  Box #3
        for (uint32_t i = 0; i < N; i++) {
            epsilon[i].resize(j + 1);
            epsilon[i][j] = sgn * (details::first_derivative<T>(t[i][j]) * (alpha[i][j] - t[i][j]) +
                details::function<T>(t[i][j]) - details::function<T>(alpha[i][j]));
        }

        epsilon[N].resize(j + 1);
        epsilon[N][j] = sgn * (details::first_derivative<T>(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) +
            details::function<T>(t[N - 1][j]) - details::function<T>(alpha[N][j]));

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = fabs(epsilon[0][j]);
        min_epsilon = fabs(epsilon[0][j]);
        for (uint32_t i = 1; i < N + 1; i++) {
            if (fabs(epsilon[i][j]) > max_epsilon) max_epsilon = fabs(epsilon[i][j]);
            if (fabs(epsilon[i][j]) < min_epsilon) min_epsilon = fabs(epsilon[i][j]);
        }

        if ((j == PWL_MAX_ITERATIONS) || (max_epsilon - min_epsilon < threshold * min_epsilon)) {
            pwl.clear();
            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (uint32_t i = 0; i < N; i++) {
                double val = sgn * details::first_derivative<T>(t[i][j]) * (alpha[i][j] - t[i][j]) +
                    sgn * details::function<T>(t[i][j]) - epsilon_final;
                double val_next = sgn * details::first_derivative<T>(t[i][j]) * (alpha[i + 1][j] - t[i][j]) +
                    sgn * details::function<T>(t[i][j]) - epsilon_final;
                double m = (val_next - val) / (alpha[i + 1][j] - alpha[i][j]);
                pwl.emplace_back(Pwl{m, val - m * alpha[i][j], alpha[i][j]});
            }
            pwl.emplace_back(Pwl{0, 0, alpha[N][j]});
            if (j == PWL_MAX_ITERATIONS) {
                std::runtime_error("Failed to converge in pivot_search!");
            }
            return epsilon_final;
        }

        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j--;
                delta /= 2;
            } else if (max_epsilon == max_epsilon_prev) {
                if (!same_epsilon) {
                    same_epsilon = true;
                } else {
                    j--;
                    delta /= 2;
                    same_epsilon = false;
                }
            }
        }

        // Figure 4:  Box #4
        for (uint32_t i = 0; i < N; i++) {
            d[i].resize(j + 1);
            d[i][j] = delta * (epsilon[i + 1][j] - epsilon[i][j]) /
                ((epsilon[i + 1][j] / (alpha[i + 1][j] - t[i][j])) + (epsilon[i][j] / (t[i][j] - alpha[i][j])));
        }

        // Figure 4:  Box #5
        for (uint32_t i = 0; i < N; i++) {
            t[i].resize(j + 2);
            t[i][j + 1] = t[i][j] + d[i][j];
        }

        t[N].resize(j + 2);
        j++;
    }
}

template<typename T>
double calculate_error_pct(
    const double lower_bound,
    const double upper_bound,
    const double offset) {
    auto samples = details::samples<T>();
    double delta = (upper_bound - lower_bound) / (samples + 1);
    double min_val = 0.0;
    double max_val = 0.0;
    if (delta < 0) {
        return 0.0;
    }

    min_val = max_val = details::function<T>(lower_bound);
    //min_val = max_val = pow(activation_type.args.pow.offset + activation_type.args.pow.scale * l_bound, activation_type.args.pow.exponent);
    for (int i = 0; i < samples; i++) {
        double arg = lower_bound + i * delta;
        double val = details::function<T>(arg);
        //val = pow(activation_type.args.pow.offset + activation_type.args.pow.scale * arg, activation_type.args.pow.exponent);
        if (val > max_val) {
            max_val = val;
        }

        if (val < min_val) {
            min_val = val;
        }
    }

    return(100.0 * fabs(offset) / (max_val - min_val));
}

template<typename T>
std::vector<Pwl> pwl_search(
    double lower_bound,
    double upper_bound,
    double threshold,
    double allowed_err_pct,
    double& err_pct) {
    std::vector<Pwl> pwl;
    if (lower_bound > upper_bound || threshold < 0) {
        return pwl;
    }

    if (split_search<T>(lower_bound, upper_bound)) {
        auto  negative_pwl = [](std::vector<Pwl>& pwl) {
            for (auto& e : pwl) {
                e.m = -e.m;
                e.b = -e.b;
            }
        };

        double err_pct1 = 0.0;
        double err_pct2 = 0.0;
        double break_bound = get_break_bound<T>();
        pwl = pwl_search<T>(lower_bound, break_bound, threshold, allowed_err_pct, err_pct1);
        negative_pwl(pwl);
        std::vector<Pwl> pwl2 = pwl_search<T>(break_bound, upper_bound, threshold, allowed_err_pct, err_pct2);
        if (std::is_same<T, opset1::Exp>::value || std::is_same<T, opset1::Power>::value) {
            negative_pwl(pwl2);
        }

        // merge
        pwl.pop_back();  // remove final alpha and beta from first half
        pwl.insert(pwl.end(), pwl2.begin(), pwl2.end());  // concatenate the two halves
        err_pct = (err_pct1 + err_pct2) / 2;  // this is not quite correct but should give an indication
    } else {
        if (std::is_same<T, opset1::Sign>::value) {
            pwl.resize(4);
            pwl[0] = {0, -1, -std::numeric_limits<float>::infinity()};
            pwl[1] = {0, 0, 0};
            pwl[2] = {0, 1, 0};
            pwl[3] = {0, 0, std::numeric_limits<float>::infinity()};
        } else if (std::is_same<T, opset1::Abs>::value) {
            pwl.resize(3);
            pwl[0] = {-1, 0, -std::numeric_limits<float>::infinity()};
            pwl[1] = {1, 0, 0};
            pwl[2] = {0, 0, std::numeric_limits<float>::infinity()};
        /*} else if (std::is_same<T, opset1::Identity>::value) {
            pwl.resize(2);
            pwl[0] = {1, 0, -std::numeric_limits<float>::infinity()};
            pwl[1] = {0, 0, std::numeric_limits<float>::infinity()};
        } else if (std::is_same<T, opset1::KaldiLstmClipping>::value) {
            pwl.resize(4);
            pwl[0] = {0, lower_bound, -std::numeric_limits<float>::infinity()};
            pwl[1] = {1, 0, lower_bound};
            pwl[2] = {0, upper_bound, uppper_bound};
            pwl[3] = {0, 0, std::numeric_limits<float>::infinity()};*/
        } else {
            bool negative = false;
            if (std::is_same<T, opset1::Sigmoid>::value ||
                std::is_same<T, opset1::Tanh>::value
                /* || std::is_same<T, opset1::SoftSign>::value*/) {
                negative = upper_bound == 0;
            } else if (std::is_same<T, opset1::Exp>::value
                //|| std::is_same<T, opset1::NegLog>::value ||
                //std::is_same<T, opset1::NegHalfLog>::value
                ) {
                negative = true;
            } else if (std::is_same<T, opset1::Power>::value) {
                //negative = (fmod(activation_type.args.pow.exponent, 1.0) == 0) ? true : false;
            }

            int n_segments_lower = 1;
            int n_segments_upper = 1;
            do {
                n_segments_lower = n_segments_upper;
                n_segments_upper *= 2;
                auto err = pivot_search<T>(pwl, n_segments_upper, lower_bound, upper_bound, threshold, negative);
                err_pct = calculate_error_pct<T>(lower_bound, upper_bound, err);
            } while (n_segments_upper < PWL_MAX_ITERATIONS && allowed_err_pct < err_pct);

            int n_segments_mid = n_segments_lower + (n_segments_upper - n_segments_lower) / 2;
            while (std::abs(n_segments_lower - n_segments_upper) > 1) {
                auto err = pivot_search<T>(pwl, n_segments_mid, lower_bound, upper_bound, threshold, negative);
                err_pct = calculate_error_pct<T>(lower_bound, upper_bound, err);
                if (allowed_err_pct == err_pct) {
                    n_segments_lower = n_segments_mid;
                    break;
                } else if (allowed_err_pct < err_pct) {
                    n_segments_lower = n_segments_mid;
                } else {
                    n_segments_upper = n_segments_mid;
                }

                n_segments_mid = n_segments_lower + (n_segments_upper - n_segments_lower) / 2;
            }

            if (n_segments_lower >= PWL_MAX_ITERATIONS) {
                std::runtime_error("Failed to converge in pwl_search!");
            }
        }
    }

    return pwl;
}

template<typename T>
bool transpose_to_pwl(const std::shared_ptr<T>& node) {
    double allowed_err_pct = 0.005;
    double err_pct = 0;
    auto segments = pwl_search<T>(
        details::lower_bound<T>(),
        details::upper_bound<T>(),
        details::threshold<T>(),
        allowed_err_pct,
        err_pct);
    if (segments.size() < 2) {
        return false;
    }

    std::vector<double> m(segments.size() - 1);
    std::vector<double> b(segments.size() - 1);
    std::vector<double> alpha(segments.size());
    for (size_t i = 0; i < segments.size() - 1; i++) {
        m[i] = segments[i].m;
        b[i] = segments[i].b;
        alpha[i] = segments[i].alpha;
    }
    alpha[segments.size() - 1] = segments[segments.size() - 1].alpha;

    auto m_constant = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size() - 1}, m);
    auto b_constant = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size() - 1}, b);
    auto alpha_constant = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size()}, alpha);
    auto pwl = std::make_shared<ngraph::opset1::Pwl>(node->input(0).get_source_output(), m_constant, b_constant, alpha_constant);
    ngraph::copy_runtime_info(node, pwl);
    replace_node(node, pwl);
    return true;
}

template<typename T>
bool transpose_to_pwl(const std::tuple<T>& args, const std::shared_ptr<Node>& node);

template<typename T, typename ...Types>
bool transpose_to_pwl(const std::tuple<T, Types...>& args, const std::shared_ptr<Node>& node) {
    auto op = std::dynamic_pointer_cast<T>(node);
    if (op) {
        return transpose_to_pwl(op);
    }
    return transpose_to_pwl<Types...>(std::tuple<Types...>(), node);
}

template<typename T>
bool transpose_to_pwl(const std::tuple<T>& args, const std::shared_ptr<Node>& node) {
    auto op = std::dynamic_pointer_cast<T>(node);
    if (op) {
        return transpose_to_pwl(op);
    }
    return false;
}

TransposeToPwl::TransposeToPwl() {
    MATCHER_SCOPE(TransposeToPwl);

    auto sigmoid = pattern::wrap_type<opset1::Sigmoid>({ pattern::any_input() });
    auto tanh = pattern::wrap_type<opset1::Tanh>({ pattern::any_input() });
    auto exp = pattern::wrap_type<opset1::Exp>({ pattern::any_input() });
    auto power = pattern::wrap_type<opset1::Power>({ pattern::any_input() });
    auto abs = pattern::wrap_type<opset1::Abs>({ pattern::any_input() });
    auto sign = pattern::wrap_type<opset1::Sign>({ pattern::any_input() });
    const auto activation_functions =
        std::make_shared<pattern::op::Or>(OutputVector{ sigmoid, tanh, exp, power, abs, sign });

    auto callback = [sigmoid, tanh, exp, power, abs, sign](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto iter = pattern_to_output.find(sigmoid);
        if (iter == pattern_to_output.end() &&
            (iter = pattern_to_output.find(tanh)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(exp)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(power)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(abs)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(sign)) == pattern_to_output.end()) {
            return false;
        }
        return transpose_to_pwl(
            std::tuple<
                opset1::Sigmoid,
                opset1::Tanh,
                opset1::Exp,
                opset1::Power,
                opset1::Abs,
                opset1::Sign>(),
            iter->second.get_node_shared_ptr());
    };

    auto m = std::make_shared<pattern::Matcher>(activation_functions, matcher_name);
    register_matcher(m, callback);
}

} // namespace pass
} // namespace ngraph
