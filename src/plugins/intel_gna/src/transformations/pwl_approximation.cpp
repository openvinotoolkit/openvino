// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pwl_approximation.hpp"

#include <iostream>
#include <memory>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <numeric>
#include <openvino/cc/ngraph/itt.hpp>
#include <vector>

#include "common/numerical_utils.hpp"
#include "ops/pwl.hpp"
#include "ops/reference/pwl.hpp"
#include "ops/util/util.hpp"
#include "transformations/utils/utils.hpp"

static constexpr double EXP_BREAK = 0.045;

using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::common;


int max_segments_number() {
    return 128;
}

//TODO remove
static size_t expected_segments_num = max_segments_number();

NGRAPH_RTTI_DEFINITION(PWLApproximation, "PWLApproximation", 0);
NGRAPH_RTTI_DEFINITION(PWLApproximationWithFq, "PWLApproximationWithFq", 0);

template <typename T>
double get_break_bound() {
    if (std::is_same<T, ngraph::opset8::Exp>::value) {
        return EXP_BREAK;
    }

    return 0;
}

template <typename T>
bool split_search(double lower_bound, double upper_bound) {
    if (lower_bound > upper_bound) {
        return false;
    }

    double break_bound = get_break_bound<T>();
    if (std::is_same<T, ngraph::opset8::Sigmoid>::value || std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset9::SoftSign>::value || std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Power>::value) {
        return lower_bound < break_bound && upper_bound > break_bound;
    }
    return false;
}

template <typename T>
double pivot_search(const details::Function<T>& activation_function,
                    std::vector<details::Pwl>& result,
                    uint32_t N,
                    double alpha_0,
                    double alpha_N,
                    bool negative,
                    double threshold = 0.1) {
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
            alpha[i][j] =
                (activation_function.get_value(t[i - 1][j]) - activation_function.get_value(t[i][j]) +
                 activation_function.first_derivative(t[i][j]) * t[i][j] -
                 activation_function.first_derivative(t[i - 1][j]) * t[i - 1][j]) /
                (activation_function.first_derivative(t[i][j]) - activation_function.first_derivative(t[i - 1][j]));
        }
        alpha[N].resize(j + 1);
        alpha[N][j] = alpha_N;

        // Figure 4:  Box #3
        for (int i = 0; i < N; i++) {
            epsilon[i].resize(j + 1);
            epsilon[i][j] = sgn * (activation_function.first_derivative(t[i][j]) * (alpha[i][j] - t[i][j]) +
                                   activation_function.get_value(t[i][j]) - activation_function.get_value(alpha[i][j]));
            if (std::isnan(epsilon[i][j])) {
                throw std::runtime_error("The value is out of range.");
            }
        }
        epsilon[N].resize(j + 1);
        epsilon[N][j] = sgn * (activation_function.first_derivative(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) +
                               activation_function.get_value(t[N - 1][j]) - activation_function.get_value(alpha[N][j]));
        if (std::isnan(epsilon[N][j])) {
            throw std::runtime_error("The value is out of range.");
        }

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = std::fabs(epsilon[0][j]);
        min_epsilon = std::fabs(epsilon[0][j]);
        for (int i = 1; i < N + 1; i++) {
            if (std::fabs(epsilon[i][j]) > max_epsilon)
                max_epsilon = std::fabs(epsilon[i][j]);
            if (std::fabs(epsilon[i][j]) < min_epsilon)
                min_epsilon = std::fabs(epsilon[i][j]);
        }
        if (j == details::max_iterations<T>() || max_epsilon - min_epsilon < threshold * min_epsilon) {
            details::Pwl value;
            result.resize(0);
            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (int i = 0; i < N; i++) {
                value.alpha = alpha[i][j];
                value.beta = sgn * activation_function.first_derivative(t[i][j]) * (value.alpha - t[i][j]) +
                             sgn * activation_function.get_value(t[i][j]) - epsilon_final;
                value.m = sgn * activation_function.first_derivative(t[i][j]);
                value.b = value.beta - value.m * value.alpha;
                result.push_back(value);
            }

            result.emplace_back(0,
                                0,
                                alpha[N][j],
                                sgn * activation_function.first_derivative(t[N - 1][j]) * (alpha[N][j] - t[N - 1][j]) +
                                    sgn * activation_function.get_value(t[N - 1][j]) - epsilon_final);
            if (j == details::max_iterations<T>()) {
                throw std::runtime_error("Failed to converge in pivot_search!");
            }
            return (epsilon_final);
        }

        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j = j - 1;
                Delta = Delta / 2;
                same_epsilon = false;
            } else if (AreFpEq(max_epsilon, max_epsilon_prev)) {
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

template <typename T>
double calculate_error_pct(const details::Function<T>& activation_function,
                           double lower_bound,
                           double upper_bound,
                           const double offset,
                           bool negative,
                           int samples = 500) {
    double delta = (upper_bound - lower_bound) / (samples + 1);
    if (delta < 0) {
        return 0.0;
    }

    double min_val = activation_function.get_value(lower_bound);
    double max_val = activation_function.get_value(lower_bound);
    for (int i = 0; i < samples; i++) {
        double arg = lower_bound + i * delta;
        double val = activation_function.get_value(arg);
        if (val > max_val)
            max_val = val;
        if (val < min_val)
            min_val = val;
    }

    double max_err = (100.0 * std::fabs(offset) / (max_val - min_val));
    return max_err;
}

template <typename T>
bool is_negative(const details::Function<T>& activation_function, double upper_bound) {
    if (std::is_same<T, ngraph::opset8::Sigmoid>::value || std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset9::SoftSign>::value) {
        return AreFpEq(upper_bound, 0.0);
    }

    if (std::is_same<T, ngraph::opset8::Exp>::value) {
        return true;
    }

    return false;
}

template <>
bool is_negative<ngraph::opset8::Power>(const details::Function<ngraph::opset8::Power>& activation_function,
                                        double upper_bound) {
    return std::fmod(activation_function.m_exponent, 1.0) == 0;
}

void negative_pwl(std::vector<details::Pwl>& data) {
    for (auto& e : data) {
        e.m = -e.m;
        e.b = -e.b;
        e.beta = -e.beta;
    }
};

template <typename T>
std::vector<details::Pwl> pwl_search(const details::Function<T>& activation_function,
                                     double lower_bound,
                                     double upper_bound) {
    if (lower_bound > upper_bound) {
        return {};
    }

    std::vector<details::Pwl> pwls;
    auto segments_number = expected_segments_num;

    if (split_search<T>(lower_bound, upper_bound)) {
        double break_bound = get_break_bound<T>();
        auto result_left_bound =
            pwl_search_count<T>(activation_function, lower_bound, break_bound, segments_number / 2 - 1);
        pwls = std::move(result_left_bound.first);
        negative_pwl(pwls);
        auto result_right_bound =
            pwl_search_count<T>(activation_function, break_bound, upper_bound, segments_number / 2 - 1);
        auto pwls_right = std::move(result_right_bound.first);
        if (std::is_same<T, ngraph::opset8::Exp>::value || std::is_same<T, ngraph::opset8::Power>::value) {
            negative_pwl(pwls_right);
        }

        // merge
        if (!pwls.empty()) {
            pwls.pop_back();  // remove final alpha from first half
        }
        pwls.insert(pwls.end(), pwls_right.begin(), pwls_right.end());  // concatenate the two halves

        // TODO remove error if it will be not finally used for validation.
        std::cout << "error1 " << result_left_bound.second << ", error2 " << result_right_bound.second << std::endl;
    } else {
        auto result = pwl_search_count(activation_function, lower_bound, upper_bound, segments_number - 2);
        pwls = std::move(result.first);
        std::cout << "error1 " << result.second << std::endl;
    }

    return pwls;
}

template <typename T>
std::pair<std::vector<details::Pwl>, double> pwl_search_count(const details::Function<T>& activation_function,
                                                              double lower_bound,
                                                              double upper_bound,
                                                              size_t max_segments) {
    std::vector<details::Pwl> pwl;
    int segments_number = max_segments;
    bool negative = is_negative<T>(activation_function, upper_bound);
    auto err = pivot_search<T>(activation_function, pwl, segments_number, lower_bound, upper_bound, negative);
    auto err_pct = calculate_error_pct<T>(activation_function, lower_bound, upper_bound, err, negative);
    return {pwl, err_pct};
}

template <typename T>
std::pair<double, double> get_bounds(const std::shared_ptr<ngraph::Node>& fake_quantize) {
    auto fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(fake_quantize);
    auto lower_bound = details::lower_bound<T>();
    auto upper_bound = details::upper_bound<T>();
    if (fq) {
        auto input_low = std::dynamic_pointer_cast<ngraph::opset8::Constant>(fq->get_input_node_shared_ptr(1));
        auto input_high = std::dynamic_pointer_cast<ngraph::opset8::Constant>(fq->get_input_node_shared_ptr(2));
        if (!ngraph_util::get_constant_value(input_low, lower_bound) ||
            !ngraph_util::get_constant_value(input_high, upper_bound)) {
            throw std::runtime_error("The unsupported type of element.");
        }

        auto abs_max = std::max(std::fabs(std::min(lower_bound, upper_bound) * 1.25),
                                std::fabs(std::max(lower_bound, upper_bound) * 1.25));
        lower_bound = abs_max < std::fabs(details::lower_bound<T>()) ? -abs_max : details::lower_bound<T>();
        upper_bound = abs_max < std::fabs(details::upper_bound<T>()) ? abs_max : details::upper_bound<T>();
    }

    return std::make_pair(lower_bound, upper_bound);
}

template <>
std::pair<double, double> get_bounds<ngraph::opset8::Log>(const std::shared_ptr<ngraph::Node>& fake_quantize) {
    return std::make_pair(details::lower_bound<ngraph::opset8::Log>(), details::upper_bound<ngraph::opset8::Log>());
}

template <typename T>
bool pwl_search(const std::shared_ptr<T>& node,
                const std::shared_ptr<ngraph::Node>& fake_quantize,
                std::vector<details::Pwl>& segments) {
    double lower_bound = 0;
    double upper_bound = 0;
    std::tie(lower_bound, upper_bound) = get_bounds<T>(fake_quantize);
    segments = pwl_search<T>(details::Function<T>(), lower_bound, upper_bound);
    if (segments.size() <= 2) {
        return false;
    }

    if (segments.front().beta < details::Function<T>::min_value()) {
        segments.front().alpha += (details::Function<T>::min_value() - segments.front().beta) / segments.front().m;
    }

    segments.insert(segments.begin(),
                    {0,
                     std::max(segments.front().beta, details::Function<T>::min_value()),
                     -std::numeric_limits<double>::infinity()});

    if (segments.back().beta > details::Function<T>::max_value()) {
        segments.back().alpha +=
            (details::Function<T>::max_value() - segments.back().beta) / segments.at(segments.size() - 2).m;
    }

    segments.back().b = std::min(segments.back().beta, details::Function<T>::max_value());
    segments.push_back({0, 0, std::numeric_limits<double>::infinity()});
    // TODO remove
    std::cout << "Segments number: " << segments.size() << std::endl;
    return true;
}

static bool pwl_search_power(const std::shared_ptr<ngraph::Node>& node,
                             double exponent,
                             double scale,
                             double offset,
                             const std::shared_ptr<ngraph::Node>& fake_quantize,
                             std::vector<details::Pwl>& segments) {
    auto fq = std::dynamic_pointer_cast<ngraph::opset8::FakeQuantize>(fake_quantize);
    auto lower_bound = details::lower_bound<ngraph::opset8::Power>(exponent);
    auto upper_bound = details::upper_bound<ngraph::opset8::Power>();
    if (fq) {
        auto output_low = std::dynamic_pointer_cast<ngraph::opset8::Constant>(fq->get_input_node_shared_ptr(1));
        auto output_high = std::dynamic_pointer_cast<ngraph::opset8::Constant>(fq->get_input_node_shared_ptr(2));
        if (!ngraph_util::get_constant_value(output_low, lower_bound) ||
            !ngraph_util::get_constant_value(output_high, upper_bound)) {
            throw std::runtime_error("The unsupported type of element.");
        }
    }

    if (AreFpEq(exponent, 1.0)) {
        // An affine primitive will be used in this case.
        return false;
    } else if (AreFpEq(exponent, 0.0)) {
        segments.emplace_back(0, 1, -std::numeric_limits<double>::infinity());
        segments.emplace_back(0, 1, std::numeric_limits<double>::infinity());
        segments.emplace_back(0, 0, std::numeric_limits<double>::infinity());
        return true;
    }

    segments = pwl_search<ngraph::opset8::Power>(details::Function<ngraph::opset8::Power>(exponent, scale, offset),
                                                 lower_bound,
                                                 upper_bound);
    if (segments.size() <= 2) {
        return false;
    }

    segments.insert(
        segments.begin(),
        {0, segments.front().beta, AreFpEq(fmod(exponent, 1.0), 0.0) ? -std::numeric_limits<double>::infinity() : 0});
    segments.back().b = segments.back().beta;
    segments.push_back({0, 0, std::numeric_limits<double>::infinity()});
    return true;
}

template <>
bool pwl_search<ngraph::opset8::Power>(const std::shared_ptr<ngraph::opset8::Power>& node,
                                       const std::shared_ptr<ngraph::Node>& fake_quantize,
                                       std::vector<details::Pwl>& segments) {
    auto constant = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node->get_input_node_shared_ptr(1));
    double exponent = 0;
    if (!ngraph_util::get_constant_value(constant, exponent)) {
        throw std::runtime_error("The unsupported type of element.");
    }

    return pwl_search_power(node, exponent, 1, 0, fake_quantize, segments);
}

template <>
bool pwl_search<ngraph::op::PowerIE>(const std::shared_ptr<ngraph::op::PowerIE>& node,
                                     const std::shared_ptr<ngraph::Node>& fake_quantize,
                                     std::vector<details::Pwl>& segments) {
    auto power = std::dynamic_pointer_cast<ngraph::op::PowerIE>(node);
    return pwl_search_power(node, power->power, power->scale, power->shift, fake_quantize, segments);
}

template <typename T>
bool transform_to_pwl(const std::shared_ptr<ngraph::Node>& fake_quantize, const std::shared_ptr<T>& node) {
    std::vector<details::Pwl> segments;
    if (!pwl_search<T>(node, fake_quantize, segments)) {
        return false;
    }

    // Final number of segments is known here
    if (segments.size() - 1 > max_segments_number()) {
        throw std::runtime_error("Failed to converge in pwl_search!");
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

    auto m_constant =
        std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64, ngraph::Shape{segments.size() - 1}, m);
    m_constant->set_friendly_name(node->get_friendly_name() + "/pwl_slope");
    auto b_constant =
        std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64, ngraph::Shape{segments.size() - 1}, b);
    b_constant->set_friendly_name(node->get_friendly_name() + "/pwl_offset");
    auto alpha_constant =
        std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::f64, ngraph::Shape{segments.size()}, alpha);
    alpha_constant->set_friendly_name(node->get_friendly_name() + "/pwl_alpha");
    auto pwl = std::make_shared<ov::intel_gna::op::Pwl>(fake_quantize ? fake_quantize : node->input_value(0),
                                                        m_constant,
                                                        b_constant,
                                                        alpha_constant);
    pwl->set_base_node(node);
    pwl->set_friendly_name(node->get_friendly_name());
    ngraph::copy_runtime_info(node, {pwl, m_constant, b_constant, alpha_constant});
    replace_node(node, pwl);
    return true;
}

static bool transform_to_pwl(std::tuple<>&&,
                             const std::shared_ptr<ngraph::Node>&,
                             const std::shared_ptr<ngraph::Node>&) {
    return false;
}

template <typename T, typename... Types>
static bool transform_to_pwl(std::tuple<T, Types...>&&,
                             const std::shared_ptr<ngraph::Node>& fake_quantize,
                             const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<T>(node);
    if (op) {
        return transform_to_pwl(fake_quantize, op);
    }
    return transform_to_pwl(std::tuple<Types...>(), fake_quantize, node);
}

static std::shared_ptr<ngraph::pattern::Matcher> create_matcher(ov::graph_rewrite_callback& handler_callback,
                                                                const std::string& matcher_name,
                                                                bool fq) {
    auto activation_input = ngraph::pattern::any_input();
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({ngraph::pattern::any_input(),
                                                                                   ngraph::pattern::any_input(),
                                                                                   ngraph::pattern::any_input(),
                                                                                   ngraph::pattern::any_input(),
                                                                                   ngraph::pattern::any_input()});
    if (fq)
        activation_input = fake_quantize;
    auto sigmoid = ngraph::pattern::wrap_type<ngraph::opset8::Sigmoid>({activation_input});
    auto tanh = ngraph::pattern::wrap_type<ngraph::opset8::Tanh>({activation_input});
    auto exp = ngraph::pattern::wrap_type<ngraph::opset8::Exp>({activation_input});
    auto power = ngraph::pattern::wrap_type<ngraph::opset8::Power>(
        {activation_input, ngraph::pattern::any_input(), ngraph::pattern::any_input()});
    auto powerIE = ngraph::pattern::wrap_type<ngraph::op::PowerIE>({activation_input});
    auto log = ngraph::pattern::wrap_type<ngraph::opset8::Log>({activation_input});
    auto softsign = ngraph::pattern::wrap_type<ngraph::opset9::SoftSign>({activation_input});
    auto activation_function =
        std::make_shared<ngraph::pattern::op::Or>(ov::OutputVector{sigmoid, tanh, exp, power, powerIE, log, softsign});

    auto callback = [=](ngraph::pattern::Matcher& m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto iter = pattern_to_output.find(sigmoid);
        if (iter == pattern_to_output.end() && (iter = pattern_to_output.find(tanh)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(exp)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(power)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(powerIE)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(log)) == pattern_to_output.end() &&
            (iter = pattern_to_output.find(softsign)) == pattern_to_output.end()) {
            return false;
        }
        auto fake_quantize_iter = pattern_to_output.find(fake_quantize);
        return transform_to_pwl(std::tuple<ngraph::opset8::Sigmoid,
                                           ngraph::opset8::Tanh,
                                           ngraph::opset8::Exp,
                                           ngraph::opset8::Power,
                                           ngraph::op::PowerIE,
                                           ngraph::opset8::Log,
                                           ngraph::opset9::SoftSign>(),
                                fake_quantize_iter != pattern_to_output.end()
                                    ? fake_quantize_iter->second.get_node_shared_ptr()
                                    : std::shared_ptr<ngraph::Node>(),
                                iter->second.get_node_shared_ptr());
    };

    handler_callback = callback;
    return std::make_shared<ngraph::pattern::Matcher>(activation_function, matcher_name);
}

// TODO remove
void set_segments(double segments) {
    auto segments_number = std::round(segments);
    if (segments_number > 1) {
        expected_segments_num = segments_number;
    }
    std::cerr << "expected segments: " << expected_segments_num << std::endl;
}

PWLApproximation::PWLApproximation(double allowed_err_pct) {
    MATCHER_SCOPE(PWLApproximation);
    ov::graph_rewrite_callback callback;
    set_segments(allowed_err_pct);
    auto m = create_matcher(callback, matcher_name, false);
    register_matcher(m, callback);
}

PWLApproximationWithFq::PWLApproximationWithFq(double allowed_err_pct) {
    MATCHER_SCOPE(PWLApproximationWithFq);
    ov::graph_rewrite_callback callback;
    set_segments(allowed_err_pct);
    auto m = create_matcher(callback, matcher_name, true);
    register_matcher(m, callback);
}
