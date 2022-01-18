// Copyright (C) 2021 Intel Corporation
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
    double m;
    double b;
    double alpha;
}; // struct Pwl

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
        return -SigmoidDomain;
    }

    static double upper_bound() {
        return SigmoidDomain;
    }

    static double threshold() {
        return 0.1;
    }

    static int samples() {
        return 500;
    }

    static constexpr double SigmoidDomain = 10;
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
        return -TanhDomain;
    }

    static double upper_bound() {
        return TanhDomain;
    }

    static double threshold() {
        return 0.1;
    }

    static int samples() {
        return 500;
    }

    static constexpr double TanhDomain = 5;
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
        return -log(INT16_MAX);
    }

    static double upper_bound() {
        return log(INT16_MAX);
    }

    static double threshold() {
        return 0.1;
    }

    static int samples() {
        return 500;
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
        return 0;
    }

    static double upper_bound() {
        return LogDomain;
    }

    static double threshold() {
        return 0.1;
    }

    static int samples() {
        return 500;
    }

    static constexpr double LogDomain = 2981;
}; // struct Function<ngraph::opset8::Log>

template<>
struct Function<ngraph::opset8::Power> {
    Function(double power, double scale, double shift) :
        m_power(power), m_scale(scale), m_shift(shift) {
    }

    static const char* name() {
        return "Power";
    }

    double get_value(double x) const {
        return pow(x * m_scale + m_shift, m_power);
    }

    double first_derivative(double x) const {
        return m_power * m_scale * pow(m_shift + x * m_scale, m_power - 1);
    }

    static double lower_bound() {
        return -PowerDomain;
    }

    static double upper_bound() {
        return PowerDomain;
    }

    static double threshold() {
        return 0.1;
    }

    static int samples() {
        return 500;
    }

    static constexpr double PowerDomain = 16.0;

    const double m_power;
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
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value>());
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
        std::is_same<T, ngraph::opset8::Sigmoid>::value>());
}

template<typename T>
double threshold(std::true_type) {
    return Function<T>::threshold();
}

template<typename T>
double threshold(std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double threshold() {
    return threshold<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Log>::value ||
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value>());
}

template<typename T>
int samples(std::true_type) {
    return Function<T>::samples();
}

template<typename T>
int samples(std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
int samples() {
    return samples<T>(std::integral_constant<bool,
        std::is_same<T, ngraph::opset8::Exp>::value ||
        std::is_same<T, ngraph::opset8::Tanh>::value ||
        std::is_same<T, ngraph::opset8::Sigmoid>::value ||
        std::is_same<T, ngraph::opset8::Power>::value ||
        std::is_same<T, ngraph::op::PowerIE>::value ||
        std::is_same<T, ngraph::opset8::Log>::value>());
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
        std::is_same<T, ngraph::opset8::Log>::value>());
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
