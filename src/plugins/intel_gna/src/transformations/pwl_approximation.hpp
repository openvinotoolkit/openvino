// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <legacy/ngraph_ops/power.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <stdexcept>
#include <transformations_visibility.hpp>
#include <vector>

#include "common/numerical_utils.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief PWLApproximation transformation replaces suitable activation function with pwl
 */
class PWLApproximation : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PWLApproximation(double max_error_percent);
};

class PWLApproximationWithFq : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    PWLApproximationWithFq(double max_error_percent);
};

namespace details {
struct Pwl {
    Pwl() = default;
    Pwl(double im, double ib, double ialpha, double ibeta = 0) : m(im), b(ib), alpha(ialpha), beta(ibeta) {}
    double m;
    double b;
    double alpha;
    double beta;
};  // struct Pwl

template <typename T>
struct Function;

template <>
struct Function<ngraph::opset8::Sigmoid> {
    static const char* name() {
        return "sigmoid";
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

    static double min_value() {
        return 0;
    }

    static double max_value() {
        return 1;
    }
};  // struct Function<ngraph::opset8::Sigmoid>

template <>
struct Function<ngraph::opset8::Tanh> {
    static const char* name() {
        return "tanh";
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

    static double min_value() {
        return -1;
    }

    static double max_value() {
        return 1;
    }
};  // struct Function<ngraph::opset8::Tanh>

template <>
struct Function<ngraph::opset8::Exp> {
    static const char* name() {
        return "exp";
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

    static double min_value() {
        return 0;
    }

    static double max_value() {
        return INT16_MAX;
    }
};  // struct Function<ngraph::opset8::Exp>

template <>
struct Function<ngraph::opset8::Log> {
    static const char* name() {
        return "log";
    }

    double get_value(double x) const {
        return std::log(x);
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

    static double min_value() {
        return -11;
    }

    static double max_value() {
        return INT16_MAX;
    }
};  // struct Function<ngraph::opset8::Log>

template <>
struct Function<ngraph::opset9::SoftSign> {
    static const char* name() {
        return "softsign";
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

    static double min_value() {
        return -1;
    }

    static double max_value() {
        return 1;
    }
};  // struct Function<ngraph::opset9::SoftSign>

template <>
struct Function<ngraph::op::PowerIE> {
    static const char* name() {
        return "power";
    }
};  // struct Function<ngraph::op::PowerIE>

template <>
struct Function<ngraph::opset8::Power> {
    Function(double exponent, double scale, double shift) : m_exponent(exponent), m_scale(scale), m_shift(shift) {}

    static const char* name() {
        return "power";
    }

    double get_value(double x) const {
        return pow(x * m_scale + m_shift, m_exponent);
    }

    double first_derivative(double x) const {
        return m_exponent * m_scale * pow(m_shift + x * m_scale, m_exponent - 1);
    }

    static double lower_bound(double exponent) {
        return common::AreFpEq(fmod(exponent, 1.0), 0.0) ? -16.0 : 0.0;
    }

    static double upper_bound() {
        return 16.0;
    }

    const double m_exponent;
    const double m_scale;
    const double m_shift;
};  // struct Function<ngraph::opset8::Power>

template <typename T>
double lower_bound(std::true_type) {
    return Function<T>::lower_bound();
}

template <typename T>
double lower_bound(std::false_type) {
    throw std::runtime_error("Not supported");
}

template <typename T>
double lower_bound() {
    return lower_bound<T>(std::integral_constant < bool,
                          std::is_same<T, ngraph::opset8::Log>::value || std::is_same<T, ngraph::opset8::Exp>::value ||
                              std::is_same<T, ngraph::opset8::Tanh>::value ||
                              std::is_same<T, ngraph::opset8::Sigmoid>::value ||
                              std::is_same<T, ngraph::opset9::SoftSign>::value > ());
}

template <typename T>
double lower_bound(double exponent, std::true_type) {
    return Function<ngraph::opset8::Power>::lower_bound(exponent);
}

template <typename T>
double lower_bound(double exponent, std::false_type) {
    throw std::runtime_error("Not supported");
}

template <typename T>
double lower_bound(double exponent) {
    return lower_bound<T>(
        exponent,
        std::integral_constant < bool,
        std::is_same<T, ngraph::opset8::Power>::value || std::is_same<T, ngraph::op::PowerIE>::value > ());
}

template <typename T>
double upper_bound(std::true_type) {
    return Function<T>::upper_bound();
}

template <typename T>
double upper_bound(std::false_type) {
    throw std::runtime_error("Not supported");
}

template <typename T>
double upper_bound() {
    return upper_bound<T>(
        std::integral_constant < bool,
        std::is_same<T, ngraph::opset8::Log>::value || std::is_same<T, ngraph::opset8::Exp>::value ||
            std::is_same<T, ngraph::opset8::Tanh>::value || std::is_same<T, ngraph::opset8::Power>::value ||
            std::is_same<T, ngraph::op::PowerIE>::value || std::is_same<T, ngraph::opset8::Sigmoid>::value ||
            std::is_same<T, ngraph::opset9::SoftSign>::value > ());
}

template <typename T>
const char* name(std::true_type) {
    return Function<T>::name();
}

template <typename T>
const char* name(std::false_type) {
    throw std::runtime_error("Not supported");
}

template <typename T>
const char* name() {
    return name<T>(std::integral_constant < bool,
                   std::is_same<T, ngraph::opset8::Exp>::value || std::is_same<T, ngraph::opset8::Tanh>::value ||
                       std::is_same<T, ngraph::opset8::Sigmoid>::value ||
                       std::is_same<T, ngraph::opset8::Power>::value || std::is_same<T, ngraph::op::PowerIE>::value ||
                       std::is_same<T, ngraph::opset8::Log>::value ||
                       std::is_same<T, ngraph::opset9::SoftSign>::value > ());
}

template <typename T>
int max_segments_number() {
    return 128;
}

template <typename T>
inline int max_iterations() {
    return 2000;
}

template <>
inline int max_iterations<ngraph::opset8::Log>() {
    return 5000;
}

}  // namespace details
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
