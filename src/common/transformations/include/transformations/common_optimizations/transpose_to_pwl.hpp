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
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief TransposeToPwl transformation replaces suitable activation function with pwl
 */
class TRANSFORMATIONS_API TransposeToPwl : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    TransposeToPwl();
};

namespace details {

template<typename T>
struct Function;

template<>
struct Function<opset1::Sigmoid> {
    static const char* name() {
        return "Sigmoid";
    }

    static double get_value(double x) {
        return 0.5 * (1.0 + tanh(x / 2.0));
    }

    static double first_derivative(double x) {
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
}; // struct Function<opset1::Sigmoid>

template<>
struct Function<opset1::Tanh> {
    static const char* name() {
        return "Tanh";
    }

    static double get_value(double x) {
        return tanh(x);
    }

    static double first_derivative(double x) {
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
}; // struct Function<opset1::Tanh>

template<>
struct Function<opset1::Exp> {
    static const char* name() {
        return "Exp";
    }

    static double get_value(double x) {
        return exp(x);
    }

    static double first_derivative(double x) {
        return exp(x);
    }

    static double lower_bound() {
        return 0;
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
}; // struct Function<opset1::Exp>

template<>
struct Function<opset1::Abs> {
    static const char* name() {
        return "Abs";
    }

    static double lower_bound() {
        return -1;
    }

    static double upper_bound() {
        return 1;
    }

    static double threshold() {
        return 0.1;
    }
}; // struct Function<opset1::Abs>

template<>
struct Function<opset1::Sign> {
    static const char* name() {
        return "Sign";
    }

    static double lower_bound() {
        return -1;
    }

    static double upper_bound() {
        return 1;
    }

    static double threshold() {
        return 0.1;
    }
}; // struct Function<opset1::Sign>

template<typename T>
double function(double x,  std::true_type) {
    return Function<T>::get_value(x);
}

template<typename T>
double function(double x,  std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double function(double x) {
    return function<T>(x, std::integral_constant<bool,
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
}

template<typename T>
double first_derivative(double x, std::true_type) {
    return Function<T>::first_derivative(x);
}

template<typename T>
double first_derivative(double x, std::false_type) {
    throw std::runtime_error("Not supported");
}

template<typename T>
double first_derivative(double x) {
    return first_derivative<T>(x, std::integral_constant<bool,
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
}

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
        std::is_same<T, opset1::Abs>::value ||
        std::is_same<T, opset1::Sign>::value ||
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
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
        std::is_same<T, opset1::Abs>::value ||
        std::is_same<T, opset1::Sign>::value ||
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
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
        std::is_same<T, opset1::Abs>::value ||
        std::is_same<T, opset1::Sign>::value ||
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
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
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
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
        std::is_same<T, opset1::Exp>::value ||
        std::is_same<T, opset1::Tanh>::value ||
        std::is_same<T, opset1::Sigmoid>::value>());
}

} // namespace details
} // namespace pass
} // namespace ngraph
