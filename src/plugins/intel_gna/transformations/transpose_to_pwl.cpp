// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_to_pwl.hpp"
#include "transformations/utils/utils.hpp"
#include "ops/pwl.hpp"

#include <memory>
#include <vector>
#include <numeric>
#include <iostream>

#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>

static constexpr double EXP_BREAK = 0.045;

namespace GNAPluginNS {

NGRAPH_RTTI_DEFINITION(TransposeToPwl, "TransposeToPwl", 0);

template<typename T>
double get_break_bound() {
    if (std::is_same<T, ngraph::opset8::Exp>::value) {
        return EXP_BREAK;
    }

    return 0;
}

template<typename T>
bool split_search(double lower_bound, double upper_bound) {
    if (lower_bound > upper_bound) {
        return false;
    }

    double break_bound = get_break_bound<T>();
    if (std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        //std::is_same<T, ngraph::opset8::SoftSign>::value ||
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Power>::value) {
        return lower_bound < break_bound && upper_bound > break_bound;
    }
    return false;
}

template<typename T>
double pivot_search(
    const details::Function<T>& activation_function,
    std::vector<details::Pwl>& pwl,
    uint32_t N,
    double alpha_0,
    double alpha_N,
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
    if (details::threshold<T>() < 0) {
        return epsilon_final;
    }

    // Figure 4:  Box #1
    int j = 0;
    double delta = 1.0;

    for (int i = 0; i < N; i++) {
        t[i].resize(2);
        t[i][1] = alpha_0 + (static_cast<double>((i + 1)) / static_cast<double>((N + 1))) * (alpha_N - alpha_0);
    }

    for (int i = 0; i <= N; i++) {
        alpha[i].resize(2);
        epsilon[i].resize(2);
        d[i].resize(2);
    }

    while (true) {
        // Figure 4:  Box #2
        alpha[0][1] = alpha_0;
        for (int i = 1; i < N; i++) {
            alpha[i][1] = (activation_function.get_value(t[i - 1][1]) - activation_function.get_value(t[i][1]) +
                    activation_function.first_derivative(t[i][1]) * t[i][1] - activation_function.first_derivative(t[i - 1][1]) * t[i - 1][1])
                / (activation_function.first_derivative(t[i][1]) - activation_function.first_derivative(t[i - 1][1]));
        }
        alpha[N][1] = alpha_N;

        // Figure 4:  Box #3
        for (int i = 0; i < N; i++) {
            epsilon[i][1] = sgn * (activation_function.first_derivative(t[i][1]) * (alpha[i][1] - t[i][1]) +
                activation_function.get_value(t[i][1]) - activation_function.get_value(alpha[i][1]));
        }

        epsilon[N][1] = sgn * (activation_function.first_derivative(t[N - 1][1]) * (alpha[N][1] - t[N - 1][1]) +
            activation_function.get_value(t[N - 1][1]) - activation_function.get_value(alpha[N][1]));

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = std::abs(epsilon[0][1]);
        min_epsilon = std::abs(epsilon[0][1]);
        for (int i = 1; i < N + 1; i++) {
            if (std::abs(epsilon[i][1]) > max_epsilon) max_epsilon = std::abs(epsilon[i][1]);
            if (std::abs(epsilon[i][1]) < min_epsilon) min_epsilon = std::abs(epsilon[i][1]);
        }

        if (j == details::max_iterations<T>() || max_epsilon - min_epsilon < details::threshold<T>() * min_epsilon) {
            details::Pwl value;
            pwl.resize(0);
            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (int i = 0; i < N; i++) {
                value.alpha = alpha[i][1];
                double val = sgn * activation_function.first_derivative(t[i][1]) * (value.alpha - t[i][1]) +
                    sgn * activation_function.get_value(t[i][1]) - epsilon_final;
                double val_next = sgn * activation_function.first_derivative(t[i][1]) * (alpha[i + 1][1] - t[i][1]) +
                    sgn * activation_function.get_value(t[i][1]) - epsilon_final;
                value.m = (val_next - val) / (alpha[i + 1][1] - value.alpha);
                value.b = (val - value.m * value.alpha);
                pwl.push_back(value);
            }
            value.m = value.b = 0.0;
            value.alpha = alpha[N][1];
            pwl.push_back(value);
            if (j == details::max_iterations<T>()) {
                throw std::runtime_error("Failed to converge in pivot_search!");
            }

            return epsilon_final;
        }

        int jj = 1;
        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j--;
                jj--;
                delta /= 2;
            } else if (max_epsilon == max_epsilon_prev) {
                if (!same_epsilon) {
                    same_epsilon = true;
                } else {
                    j--;
                    jj--;
                    delta /= 2;
                    same_epsilon = false;
                }
            }
        }

        // Figure 4:  Box #4
        for (int i = 0; i < N; i++) {
            d[i][jj] = delta * (epsilon[i + 1][jj] - epsilon[i][jj]) /
                ((epsilon[i + 1][jj] / (alpha[i + 1][jj] - t[i][jj])) + (epsilon[i][jj] / (t[i][jj] - alpha[i][jj])));
        }

        // Figure 4:  Box #5
        for (int i = 0; i < N; i++) {
            double tmp = t[i][jj];
            t[i][1] = t[i][jj] + d[i][jj];
            t[i][0] = tmp;
        }

        j++;

        if (jj == 1) {
            for (int i = 0; i <= N; i++) {
                alpha[i][0] = alpha[i][1];
                epsilon[i][0] = epsilon[i][1];
                d[i][0] = d[i][1];
            }
        }
    }
}

template<typename T>
double calculate_error_pct(const details::Function<T>& activation_function,
                           double lower_bound,
                           double upper_bound,
                           const double offset) {
    auto samples = details::samples<T>();
    double delta = (upper_bound - lower_bound) / (samples + 1);
    double min_val = 0.0;
    double max_val = 0.0;
    if (delta < 0) {
        return 0.0;
    }

    min_val = max_val = activation_function.get_value(lower_bound);
    for (int i = 0; i < samples; i++) {
        double val = activation_function.get_value(lower_bound + i * delta);
        if (val > max_val) {
            max_val = val;
        }

        if (val < min_val) {
            min_val = val;
        }
    }

    return(100.0 * std::abs(offset) / (max_val - min_val));
}

template<typename T>
bool is_negative(const details::Function<T>& activation_function, double upper_bound) {
    if (std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value
        /* || std::is_same<T, ngraph::opset8::SoftSign>::value*/) {
        return upper_bound == 0;
    }

    if (std::is_same<T, ngraph::opset8::Exp>::value
       //std::is_same<T, ngraph::opset8::NegLog>::value ||
       //std::is_same<T, ngraph::opset8::NegHalfLog>::value
       ) {
        return true;
    }

    return false;
}

template<>
bool is_negative<ngraph::opset8::Power>(const details::Function<ngraph::opset8::Power>& activation_function, double upper_bound) {
    return fmod(activation_function.m_power, 1.0) == 0;
}

template<typename T>
std::vector<details::Pwl> pwl_search(const details::Function<T>& activation_function,
                                     double lower_bound,
                                     double upper_bound,
                                     double allowed_err_pct,
                                     double& err_pct) {
    std::vector<details::Pwl> pwl;
    if (lower_bound > upper_bound || details::threshold<T>() < 0) {
        return pwl;
    }

    if (split_search<T>(lower_bound, upper_bound)) {
        auto negative_pwl = [](std::vector<details::Pwl>& data) {
            for (auto& e : data) {
                e.m = -e.m;
                e.b = -e.b;
            }
        };

        double err_pct1 = 0.0;
        double err_pct2 = 0.0;
        double break_bound = get_break_bound<T>();
        pwl = pwl_search<T>(activation_function, lower_bound, break_bound, allowed_err_pct, err_pct1);
        negative_pwl(pwl);
        std::vector<details::Pwl> pwl2 = pwl_search<T>(activation_function, break_bound, upper_bound, allowed_err_pct, err_pct2);
        if (std::is_same<T, ngraph::opset8::Exp>::value ||
            std::is_same<T, ngraph::opset8::Power>::value) {
            negative_pwl(pwl2);
        }

        // merge
        if (!pwl.empty())
            pwl.pop_back();  // remove final alpha from first half
        pwl.insert(pwl.end(), pwl2.begin(), pwl2.end());  // concatenate the two halves
        err_pct = (err_pct1 + err_pct2) / 2;  // this is not quite correct but should give an indication
    } else {
        int segments_number = 1;
        bool negative = is_negative<T>(activation_function, upper_bound);
        auto err = pivot_search<T>(activation_function, pwl, segments_number, lower_bound, upper_bound, negative);
        err_pct = calculate_error_pct<T>(activation_function, lower_bound, upper_bound, err);
        while (segments_number < details::max_segments_number<T>() && allowed_err_pct < err_pct) {
            segments_number++;
            err = pivot_search<T>(activation_function, pwl, segments_number, lower_bound, upper_bound, negative);
            err_pct = calculate_error_pct<T>(activation_function, lower_bound, upper_bound, err);
        }

        if (segments_number >= details::max_segments_number<T>()) {
            throw std::runtime_error("Failed to converge in pwl_search!");
        }
    }

    return pwl;
}

template<typename T>
std::vector<details::Pwl> pwl_search(const std::shared_ptr<T>& node,
                                     double allowed_err_pct,
                                     double& err_pct) {
    return pwl_search<T>(details::Function<T>(),
                         details::lower_bound<T>(),
                         details::upper_bound<T>(),
                         allowed_err_pct,
                         err_pct);
}

template<>
std::vector<details::Pwl> pwl_search<ngraph::opset8::Power>(const std::shared_ptr<ngraph::opset8::Power>& node,
                                                            double allowed_err_pct,
                                                            double& err_pct) {
    auto power = std::dynamic_pointer_cast<ngraph::opset8::Power>(node);
    return pwl_search<ngraph::opset8::Power>(details::Function<ngraph::opset8::Power>(0, 0, 0),
                                             details::lower_bound<ngraph::opset8::Power>(),
                                             details::upper_bound<ngraph::opset8::Power>(),
                                             allowed_err_pct,
                                             err_pct);
}

template<>
std::vector<details::Pwl> pwl_search<ngraph::op::PowerIE>(const std::shared_ptr<ngraph::op::PowerIE>& node,
                                                          double allowed_err_pct,
                                                          double& err_pct) {
    auto power = std::dynamic_pointer_cast<ngraph::op::PowerIE>(node);
    return pwl_search<ngraph::opset8::Power>(details::Function<ngraph::opset8::Power>(power->power, power->scale, power->shift),
                                             details::lower_bound<ngraph::opset8::Power>(),
                                             details::upper_bound<ngraph::opset8::Power>(),
                                             allowed_err_pct,
                                             err_pct);
}

template<typename T>
bool transpose_to_pwl(const std::shared_ptr<T>& node, double allowed_err_pct) {
    double err_pct = 0;
    auto segments = pwl_search<T>(node, allowed_err_pct, err_pct);
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

    auto m_constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size() - 1}, m);
    auto b_constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size() - 1}, b);
    auto alpha_constant = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64,
        ngraph::Shape{segments.size()}, alpha);
    auto pwl = std::make_shared<Pwl>(node->input(0).get_source_output(), m_constant, b_constant, alpha_constant);
    pwl->set_friendly_name(node->get_friendly_name());
    ngraph::copy_runtime_info(node, pwl);
    replace_node(node, pwl);
    return true;
}

template<typename T>
bool transpose_to_pwl(const std::tuple<T>& args,
                      const std::shared_ptr<ngraph::Node>& node,
                      double allowed_err_pct);

template<typename T, typename ...Types>
bool transpose_to_pwl(const std::tuple<T, Types...>& args,
                      const std::shared_ptr<ngraph::Node>& node,
                      double allowed_err_pct) {
    auto op = std::dynamic_pointer_cast<T>(node);
    if (op) {
        return transpose_to_pwl(op, allowed_err_pct);
    }
    return transpose_to_pwl<Types...>(std::tuple<Types...>(), node, allowed_err_pct);
}

template<typename T>
bool transpose_to_pwl(const std::tuple<T>& args,
                      const std::shared_ptr<ngraph::Node>& node,
                      double allowed_err_pct) {
    auto op = std::dynamic_pointer_cast<T>(node);
    if (op) {
        return transpose_to_pwl(op, allowed_err_pct);
    }
    return false;
}

TransposeToPwl::TransposeToPwl(double allowed_err_pct) {
    MATCHER_SCOPE(TransposeToPwl);
    auto sigmoid = ngraph::pattern::wrap_type<ngraph::opset8::Sigmoid>({ ngraph::pattern::any_input() });
    auto tanh = ngraph::pattern::wrap_type<ngraph::opset8::Tanh>({ ngraph::pattern::any_input() });
    auto exp = ngraph::pattern::wrap_type<ngraph::opset8::Exp>({ ngraph::pattern::any_input() });
    auto power = ngraph::pattern::wrap_type<ngraph::opset8::Power>({ ngraph::pattern::any_input(), ngraph::pattern::any_input() });
    auto powerIE = ngraph::pattern::wrap_type<ngraph::op::PowerIE>({ ngraph::pattern::any_input() });
    auto log = ngraph::pattern::wrap_type<ngraph::opset8::Log>({ ngraph::pattern::any_input() });
    const auto activation_functions =
        std::make_shared<ngraph::pattern::op::Or>(ov::OutputVector{ sigmoid, tanh, exp, power, powerIE, log });

    auto callback = [sigmoid,
                     tanh,
                     exp,
                     power,
                     powerIE,
                     log,
                     allowed_err_pct](ngraph::pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto iter = pattern_to_output.find(sigmoid);
        if (iter == pattern_to_output.end() &&
            (iter = pattern_to_output.find(tanh)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(exp)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(power)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(powerIE)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(log)) == pattern_to_output.end()) {
            return false;
        }
        return transpose_to_pwl(
            std::tuple<
                ngraph::opset8::Sigmoid,
                ngraph::opset8::Tanh,
                ngraph::opset8::Exp,
                ngraph::opset8::Power,
                ngraph::op::PowerIE,
                ngraph::opset8::Log>(),
            iter->second.get_node_shared_ptr(),
            allowed_err_pct);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(activation_functions, matcher_name);
    register_matcher(m, callback);
}

} // namespace GNAPluginNS
