// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <transformations/init_node_info.hpp>
#include <type_traits>

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/pwl_approximation.hpp"

using namespace ov::intel_gna::common;

namespace {

template <typename T>
struct Function {};

template <>
struct Function<ngraph::opset9::Sigmoid> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return 0.5 * (1.0 + std::tanh(x / 2.0));
        };
    }
};

template <>
struct Function<ngraph::opset9::Tanh> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::tanh(x);
        };
    }
};

template <>
struct Function<ngraph::opset9::SoftSign> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return x / (1.0 + std::abs(x));
        };
    }
};

template <>
struct Function<ngraph::opset9::Log> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::log(x);
        };
    }
};

template <>
struct Function<ngraph::opset9::Exp> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::exp(x);
        };
    }
};

template <>
struct Function<ngraph::opset9::Power> {
    static std::function<double(double)> get_function(double exp) {
        return [exp](const double x) {
            return std::pow(x, exp);
        };
    }
};

template <typename T>
using Enable =
    std::enable_if<std::is_same<T, ngraph::opset9::Sigmoid>::value || std::is_same<T, ngraph::opset9::Tanh>::value ||
                       std::is_same<T, ngraph::opset9::SoftSign>::value ||
                       std::is_same<T, ngraph::opset9::Log>::value || std::is_same<T, ngraph::opset9::Exp>::value,
                   int>;
template <typename T>
using EnableWithExtraArg = std::enable_if<std::is_same<T, ngraph::opset9::Power>::value, int>;

template <typename T>
class GnaPWlTestsFixture {
public:
    template <typename U = T, typename Enable<U>::type = 0>
    GnaPWlTestsFixture(const ngraph::Shape& input_shape,
                       double lower_bound,
                       double upper_bound,
                       double max_error_percent);

    template <typename U = T, typename EnableWithExtraArg<U>::type = 0>
    GnaPWlTestsFixture(const ngraph::Shape& input_shape,
                       double lower_bound,
                       double upper_bound,
                       double exp,
                       double max_error_percent);

    void run();

private:
    void validate_results(const std::vector<float>& input_data,
                          const std::vector<float>& results,
                          double max_error_percent);

    double count_abs_peak_to_peak(int samples = 1000);

    template <typename U = T>
    static std::shared_ptr<ngraph::Function> create_activation_function(const ngraph::Shape& input_shape);

    template <typename U = T>
    static std::shared_ptr<ngraph::Function> create_activation_function(const ngraph::Shape& input_shape, double exp);

    double _lower_bound;
    double _upper_bound;
    double _max_error_percent;
    std::shared_ptr<ngraph::Function> _function_under_test;
    std::function<double(double)> _reference_function;
};

template <typename T>
template <typename U, typename Enable<U>::type>
inline GnaPWlTestsFixture<T>::GnaPWlTestsFixture(const ngraph::Shape& input_shape,
                                                 double lower_bound,
                                                 double upper_bound,
                                                 double max_error_percent)
    : _lower_bound(lower_bound),
      _upper_bound(upper_bound),
      _max_error_percent(max_error_percent) {
    _function_under_test = create_activation_function<U>(input_shape);
    _reference_function = Function<U>::get_function();
}

template <typename T>
template <typename U, typename EnableWithExtraArg<U>::type>
inline GnaPWlTestsFixture<T>::GnaPWlTestsFixture(const ngraph::Shape& input_shape,
                                                 double lower_bound,
                                                 double upper_bound,
                                                 double exp,
                                                 double max_error_percent)
    : _lower_bound(lower_bound),
      _upper_bound(upper_bound),
      _max_error_percent(max_error_percent) {
    _function_under_test = create_activation_function<U>(input_shape, exp);
    _reference_function = Function<U>::get_function(exp);
}

template <typename T>
template <typename U>
inline std::shared_ptr<ngraph::Function> GnaPWlTestsFixture<T>::create_activation_function(
    const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    auto f = std::make_shared<T>(input_params);
    auto result = std::make_shared<ngraph::opset8::Result>(f);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

template <typename T>
template <typename U>
inline std::shared_ptr<ngraph::Function> GnaPWlTestsFixture<T>::create_activation_function(
    const ngraph::Shape& input_shape,
    double exp) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    auto exponents = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {exp});
    auto f = std::make_shared<T>(input_params, exponents);
    auto result = std::make_shared<ngraph::opset8::Result>(f);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

template <typename T>
inline double GnaPWlTestsFixture<T>::count_abs_peak_to_peak(int samples) {
    double delta = (_upper_bound - _lower_bound) / (samples + 1);

    if (delta <= 0) {
        std::stringstream str_stream;
        str_stream << "Cannot count test parameters for given data!! Lower bound=" << _lower_bound
                   << ", upper bound=" << _upper_bound << std::endl;
        throw std::runtime_error(str_stream.str());
    }

    double min_val = _reference_function(_lower_bound);
    double max_val = min_val;

    for (int i = 0; i < samples; i++) {
        double arg = _lower_bound + i * delta;
        double val = _reference_function(arg);
        if (val > max_val)
            max_val = val;
        if (val < min_val)
            min_val = val;
    }

    return std::abs(max_val - min_val);
}

template <typename T>
inline void GnaPWlTestsFixture<T>::run() {
    {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::PWLApproximation>(_max_error_percent);
        m.run_passes(_function_under_test);
        ASSERT_NO_THROW(check_rt_info(_function_under_test));
    }

    auto shape = _function_under_test->input().get_node_shared_ptr()->get_output_shape(0);
    ov::runtime::TensorVector result(1);
    std::vector<float> data =
        ov::test::utils::generate_float_numbers(ov::shape_size(shape), _lower_bound, _upper_bound);
    ov::runtime::Tensor input{ov::element::f32, shape, data.data()};
    ASSERT_TRUE(_function_under_test->evaluate(result, ov::runtime::TensorVector{input}));

    const float* result_data = result[0].data<float>();
    std::vector<float> results(result_data, result_data + result[0].get_size());

    validate_results(data, results, _max_error_percent);
}

template <typename T>
inline void GnaPWlTestsFixture<T>::validate_results(const std::vector<float>& input_data,
                                                    const std::vector<float>& results,
                                                    double max_error_percent) {
    ASSERT_FALSE(results.empty());

    std::vector<float> reference_values;
    std::for_each(input_data.begin(), input_data.end(), [&reference_values, this](const double& x) {
        reference_values.push_back(_reference_function(x));
    });

    ASSERT_EQ(results.size(), reference_values.size());

    auto abs_peak_to_peak = count_abs_peak_to_peak();

    for (int i = 0; i < results.size(); ++i) {
        double delta = std::abs(static_cast<double>(results[i]) - static_cast<double>(reference_values[i]));
        double error_percentage = delta / abs_peak_to_peak * 100.0;
        EXPECT_TRUE(error_percentage < max_error_percent);
    }
}

TEST(GnaPwlTest, Sigmoid) {
    GnaPWlTestsFixture<ngraph::opset9::Sigmoid> test_instance({1, 100}, -10.0, 10.0, 1.0);
    test_instance.run();
}

TEST(GnaPwlTest, Tanh) {
    GnaPWlTestsFixture<ngraph::opset9::Tanh> test_instance({1, 32}, -5.0, 5.0, 1.0);
    test_instance.run();
}

TEST(GnaPwlTest, Exp) {
    GnaPWlTestsFixture<ngraph::opset9::Exp> test_instance({1, 32}, -std::log2(INT16_MAX), std::log10(INT16_MAX), 1.0);
    test_instance.run();
}

TEST(GnaPwlTest, SoftSign) {
    GnaPWlTestsFixture<ngraph::opset9::SoftSign> test_instance({1, 32}, -10, 10, 1.0);
    test_instance.run();
}

TEST(GnaPwlTest, Log) {
    GnaPWlTestsFixture<ngraph::opset9::Log> test_instance({1, 32}, 0.001, 2981, 1.0);
    test_instance.run();
}

TEST(GnaPwlTest, Power) {
    for (float exp = 1; exp <= 2.2; exp += 0.1) {
        GnaPWlTestsFixture<ngraph::opset9::Power> test_instance({1, 32},
                                                                AreFpEq(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
                                                                16,
                                                                exp,
                                                                1.0);
        test_instance.run();
    }
}
}  // namespace
