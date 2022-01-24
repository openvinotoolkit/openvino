// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "ngraph/pattern/matcher.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <legacy/ngraph_ops/power.hpp>

#include "ops/softsign.hpp"

namespace GNAPluginNS {
/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToPwl transformation replaces suitable activation function with pwl
 */
class TRANSFORMATIONS_API TransposeToPwl : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeToPwl(double max_error_percent);
};

namespace details {
struct Pwl {
    Pwl() = default;
    Pwl(double im, double ib, double ialpha) : m(im), b(ib), alpha(ialpha) {}
    double m;
    double b;
    double alpha;
}; // struct Pwl

inline bool are_floats_equal(float p1, float p2) {
    return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
}

template<typename T>
struct Function;

template<>
struct Function<ngraph::opset8::Sigmoid> {
    static const char* name() {
        return "Sigmoid";
    }

    double get_value(double x) const {
        return 0.5 * (1.0 + tanh(x / 2.0));
    }

    double first_derivative(double x) const {
        return get_value(x) * (1.0 - get_value(x));
    }

    static double lower_bound() {
        return -10;
    }

    static double upper_bound() {
        return 10;
    }
}; // struct Function<ngraph::opset8::Sigmoid>

template<>
struct Function<ngraph::opset8::Tanh> {
    static const char* name() {
        return "Tanh";
    }

    double get_value(double x) const {
        return tanh(x);
    }

    double first_derivative(double x) const {
        return 1.0 - tanh(x) * tanh(x);
    }

    static double lower_bound() {
        return -5;
    }

    static double upper_bound() {
        return 5;
    }
}; // struct Function<ngraph::opset8::Tanh>

template<>
struct Function<ngraph::opset8::Exp> {
    static const char* name() {
        return "Exp";
    }

    double get_value(double x) const {
        return exp(x);
    }

    double first_derivative(double x) const {
        return exp(x);
    }

    static double lower_bound() {
        return -std::log2(INT16_MAX);
    }

    static double upper_bound() {
        return std::log10(INT16_MAX);
    }
}; // struct Function<ngraph::opset8::Exp>

template<>
struct Function<ngraph::opset8::Log> {
    static const char* name() {
        return "Log";
    }

    double get_value(double x) const {
        return log(x);
    }

    double first_derivative(double x) const {
        return 1.0 / x;
    }

    static double lower_bound() {
        return 0.001;
    }

    static double upper_bound() {
        return 2981;
    }
}; // struct Function<ngraph::opset8::Log>

template<>
struct Function<SoftSign> {
    static const char* name() {
        return "SoftSign";
    }

    double get_value(double x) const {
        return x / (1.0 + std::abs(x));
    }

    double first_derivative(double x) const {
        return 1.0 / ((1.0 + std::abs(x)) * (1.0 + std::abs(x)));
    }

    static double lower_bound() {
        return -10;
    }

    static double upper_bound() {
        return 10;
    }
}; // struct Function<SoftSign>

template<>
struct Function<ngraph::opset8::Power> {
    Function(double exponent, double scale, double shift) :
        m_exponent(exponent), m_scale(scale), m_shift(shift) {
    }

    static const char* name() {
        return "Power";
    }

    double get_value(double x) const {
        return pow(x * m_scale + m_shift, m_exponent);
    }

    double first_derivative(double x) const {
        return m_exponent * m_scale * pow(m_shift + x * m_scale, m_exponent - 1);
    }

    static double lower_bound(double exponent) {
        return std::max(are_floats_equal(std::fmod(exponent, 1.0), 0.0) ? static_cast<double>(std::numeric_limits<int32_t>::min()) : 0., -16.);
    }

    static double upper_bound() {
        return std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), 16.);
    }

    const double m_exponent;
    const double m_scale;
    const double m_shift;
}; // struct Function<ngraph::opset8::Power>

template<typename T>
double lower_bound(std::true_type) {
    return Function<T>::lower_bound();
}

template<typename T>
double lower_bound(std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double lower_bound() {
    return lower_bound<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Log>::value ||
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, SoftSign>::value>());
}

template<typename T>
double lower_bound(std::true_type, double exponent) {
    return Function<ngraph::opset8::Power>::lower_bound(exponent);
}

template<typename T>
double lower_bound(std::false_type, double exponent) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double lower_bound(double exponent) {
    return lower_bound<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value>(), exponent);
}

template<typename T>
double upper_bound(std::true_type) {
    return Function<T>::upper_bound();
}

template<typename T>
double upper_bound(std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double upper_bound() {
    return upper_bound<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Log>::value ||
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, SoftSign>::value>());
}

template<typename T>
const char* name(std::true_type) {
    return Function<T>::name();
}

template<typename T>
const char* name(std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
const char* name() {
    return name<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value ||
        std::is_same<T, ngraph::opset8::Log>::value ||
        std::is_same<T, SoftSign>::value>());
}

template<typename T>
int max_segments_number() {
    return 128;
}

template<typename T>
inline int max_iterations() {
    return 2000;
}

template<>
inline int max_iterations<ngraph::opset8::Log>() {
    return 5000;
}

} // namespace details
} // namespace GNAPluginNS
