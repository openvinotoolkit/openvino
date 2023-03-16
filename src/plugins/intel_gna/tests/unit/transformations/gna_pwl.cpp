// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <sstream>
#include <stdexcept>
#include <transformations/init_node_info.hpp>
#include <type_traits>

#include "common/numerical_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"
#include "transformations/pwl_approximation.hpp"

using namespace ov::intel_gna;
using namespace ov::intel_gna::common;

constexpr double kAccuracyMaxErrorPrecentage = 0.09;
constexpr double kPerformanceMaxErrorPrecentage = 1.0;

double get_max_error_percentage(const PWLApproximationMode& mode) {
    if (PWLApproximationMode::ACCURACY == mode) {
        return kAccuracyMaxErrorPrecentage;
    }
    return kPerformanceMaxErrorPrecentage;
}

namespace {

template <typename T>
struct Function {};

template <>
struct Function<ov::opset11::Sigmoid> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return 0.5 * (1.0 + std::tanh(x / 2.0));
        };
    }
};

template <>
struct Function<ov::opset11::Tanh> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::tanh(x);
        };
    }
};

template <>
struct Function<ov::opset11::SoftSign> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return x / (1.0 + std::abs(x));
        };
    }
};

template <>
struct Function<ov::opset11::Log> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::log(x);
        };
    }
};

template <>
struct Function<ov::opset11::Exp> {
    static std::function<double(double)> get_function() {
        return [](const double x) {
            return std::exp(x);
        };
    }
};

template <>
struct Function<ov::opset11::Power> {
    static std::function<double(double)> get_function(double exp) {
        return [exp](const double x) {
            return std::pow(x, exp);
        };
    }
};

template <typename T>
using Enable =
    std::enable_if<std::is_same<T, ov::opset11::Sigmoid>::value || std::is_same<T, ov::opset11::Tanh>::value ||
                       std::is_same<T, ov::opset11::SoftSign>::value || std::is_same<T, ov::opset11::Log>::value ||
                       std::is_same<T, ov::opset11::Exp>::value,
                   int>;
template <typename T>
using EnableWithExtraArg = std::enable_if<std::is_same<T, ov::opset11::Power>::value, int>;

template <typename T>
class GnaPWlTestsFixture {
public:
    template <typename U = T, typename Enable<U>::type = 0>
    GnaPWlTestsFixture(const ov::Shape& input_shape,
                       double lower_bound,
                       double upper_bound,
                       const ov::intel_gna::PWLApproximationMode& mode,
                       bool use_fq = false);

    template <typename U = T, typename EnableWithExtraArg<U>::type = 0>
    GnaPWlTestsFixture(const ov::Shape& input_shape,
                       double lower_bound,
                       double upper_bound,
                       double exp,
                       const ov::intel_gna::PWLApproximationMode& mode,
                       bool use_fq = false);

    void run();

private:
    void validate_results(const std::vector<float>& input_data,
                          const std::vector<float>& results,
                          double allowed_error_percentage);

    double count_abs_peak_to_peak(int samples = 1000);

    template <typename U = T>
    static std::shared_ptr<ov::Model> create_activation_function(const ov::Shape& input_shape);

    template <typename U = T>
    static std::shared_ptr<ov::Model> create_activation_function_with_fq(const ov::Shape& input_shape,
                                                                         double lower_bound,
                                                                         double upper_bound);

    template <typename U = T>
    static std::shared_ptr<ov::Model> create_activation_function(const ov::Shape& input_shape, double exp);

    template <typename U = T>
    static std::shared_ptr<ov::Model> create_activation_function_with_fq(const ov::Shape& input_shape,
                                                                         double exp,
                                                                         double lower_bound,
                                                                         double upper_bound);

    double m_lower_bound;
    double m_upper_bound;
    ov::intel_gna::PWLApproximationMode m_mode;
    std::shared_ptr<ov::Model> m_function_under_test;
    std::function<double(double)> m_reference_function;
};

template <typename T>
template <typename U, typename Enable<U>::type>
inline GnaPWlTestsFixture<T>::GnaPWlTestsFixture(const ov::Shape& input_shape,
                                                 double lower_bound,
                                                 double upper_bound,
                                                 const ov::intel_gna::PWLApproximationMode& mode,
                                                 bool use_fq)
    : m_lower_bound(lower_bound),
      m_upper_bound(upper_bound),
      m_mode(mode) {
    if (use_fq) {
        m_function_under_test = create_activation_function_with_fq<U>(input_shape, m_lower_bound, m_upper_bound);
    } else {
        m_function_under_test = create_activation_function<U>(input_shape);
    }
    m_reference_function = Function<U>::get_function();
}

template <typename T>
template <typename U, typename EnableWithExtraArg<U>::type>
inline GnaPWlTestsFixture<T>::GnaPWlTestsFixture(const ov::Shape& input_shape,
                                                 double lower_bound,
                                                 double upper_bound,
                                                 double exp,
                                                 const ov::intel_gna::PWLApproximationMode& mode,
                                                 bool use_fq)
    : m_lower_bound(lower_bound),
      m_upper_bound(upper_bound),
      m_mode(mode) {
    if (use_fq) {
        m_function_under_test = create_activation_function_with_fq<U>(input_shape, exp, m_lower_bound, m_upper_bound);
    } else {
        m_function_under_test = create_activation_function<U>(input_shape, exp);
    }
    m_reference_function = Function<U>::get_function(exp);
}

template <typename T>
template <typename U>
inline std::shared_ptr<ov::Model> GnaPWlTestsFixture<T>::create_activation_function(const ov::Shape& input_shape) {
    auto input_params = std::make_shared<ov::opset11::Parameter>(ov::element::f32, input_shape);
    auto f = std::make_shared<T>(input_params);
    auto result = std::make_shared<ov::opset11::Result>(f);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

template <typename T>
template <typename U>
inline std::shared_ptr<ov::Model> GnaPWlTestsFixture<T>::create_activation_function_with_fq(
    const ov::Shape& input_shape,
    double lower_bound,
    double upper_bound) {
    auto type = ov::element::f32;
    auto input_params = std::make_shared<ov::opset11::Parameter>(type, input_shape);
    auto fq = ngraph::builder::makeFakeQuantize(input_params,
                                                type,
                                                std::numeric_limits<uint16_t>::max(),
                                                {1u},
                                                {static_cast<float>(lower_bound)},
                                                {static_cast<float>(upper_bound)},
                                                {static_cast<float>(lower_bound)},
                                                {static_cast<float>(upper_bound)});
    auto f = std::make_shared<T>(fq);
    auto result = std::make_shared<ov::opset11::Result>(f);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

template <typename T>
template <typename U>
inline std::shared_ptr<ov::Model> GnaPWlTestsFixture<T>::create_activation_function(const ov::Shape& input_shape,
                                                                                    double exp) {
    auto input_params = std::make_shared<ov::opset11::Parameter>(ov::element::f32, input_shape);
    auto exponents = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {exp});
    auto f = std::make_shared<T>(input_params, exponents);
    auto result = std::make_shared<ov::opset11::Result>(f);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

template <typename T>
template <typename U>
inline std::shared_ptr<ov::Model> GnaPWlTestsFixture<T>::create_activation_function_with_fq(
    const ov::Shape& input_shape,
    double exp,
    double lower_bound,
    double upper_bound) {
    auto type = ov::element::f32;
    auto input_params = std::make_shared<ov::opset11::Parameter>(type, input_shape);
    auto fq = ngraph::builder::makeFakeQuantize(input_params,
                                                type,
                                                std::numeric_limits<uint16_t>::max(),
                                                {1u},
                                                {static_cast<float>(lower_bound)},
                                                {static_cast<float>(upper_bound)},
                                                {static_cast<float>(lower_bound)},
                                                {static_cast<float>(upper_bound)});
    auto exponents = ov::opset11::Constant::create(ov::element::f32, ov::Shape{}, {exp});
    auto f = std::make_shared<T>(fq, exponents);
    auto result = std::make_shared<ov::opset11::Result>(f);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
}

template <typename T>
inline double GnaPWlTestsFixture<T>::count_abs_peak_to_peak(int samples) {
    double delta = (m_upper_bound - m_lower_bound) / (samples - 1);

    if (delta <= 0) {
        std::stringstream str_stream;
        str_stream << "Cannot count test parameters for given data!! Lower bound=" << m_lower_bound
                   << ", upper bound=" << m_upper_bound << std::endl;
        throw std::runtime_error(str_stream.str());
    }

    double min_val = m_reference_function(m_lower_bound);
    double max_val = min_val;

    for (int i = 0; i < samples; i++) {
        double arg = m_lower_bound + i * delta;
        double val = m_reference_function(arg);
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
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::PWLApproximation>(m_mode);
        m.run_passes(m_function_under_test);
        ASSERT_NO_THROW(check_rt_info(m_function_under_test));
    }

    auto shape = m_function_under_test->input().get_node_shared_ptr()->get_output_shape(0);
    ov::runtime::TensorVector result(1);
    std::vector<float> data =
        CommonTestUtils::generate_float_numbers(ov::shape_size(shape), m_lower_bound, m_upper_bound);

    ov::runtime::Tensor input{ov::element::f32, shape, data.data()};
    ASSERT_TRUE(m_function_under_test->evaluate(result, ov::runtime::TensorVector{input}));

    const float* result_data = result[0].data<float>();
    std::vector<float> results(result_data, result_data + result[0].get_size());

    validate_results(data, results, get_max_error_percentage(m_mode));
}

template <typename T>
inline void GnaPWlTestsFixture<T>::validate_results(const std::vector<float>& input_data,
                                                    const std::vector<float>& results,
                                                    double allowed_error_percentage) {
    ASSERT_FALSE(results.empty());

    std::vector<float> reference_values;
    std::for_each(input_data.begin(), input_data.end(), [&reference_values, this](const double& x) {
        reference_values.push_back(m_reference_function(x));
    });

    ASSERT_EQ(results.size(), reference_values.size());

    auto abs_peak_to_peak = count_abs_peak_to_peak();

    for (int i = 0; i < results.size(); ++i) {
        double delta = std::abs(static_cast<double>(results[i]) - static_cast<double>(reference_values[i]));
        double error_percentage = delta / abs_peak_to_peak * 100.0;
        EXPECT_TRUE(error_percentage < allowed_error_percentage);
    }
}

TEST(GnaPwlTest, Sigmoid) {
    GnaPWlTestsFixture<ov::opset11::Sigmoid> test_instance({1, 128},
                                                           -10.0,
                                                           10.0,
                                                           ov::intel_gna::PWLApproximationMode::ACCURACY);
    test_instance.run();
}

TEST(GnaPwlTest, Tanh) {
    GnaPWlTestsFixture<ov::opset11::Tanh> test_instance({1, 128},
                                                        -5.0,
                                                        5.0,
                                                        ov::intel_gna::PWLApproximationMode::ACCURACY);
    test_instance.run();
}

TEST(GnaPwlTest, Exp) {
    GnaPWlTestsFixture<ov::opset11::Exp> test_instance({1, 128},
                                                       -std::log2(INT16_MAX),
                                                       std::log10(INT16_MAX),
                                                       ov::intel_gna::PWLApproximationMode::ACCURACY);
    test_instance.run();
}

TEST(GnaPwlTest, SoftSign) {
    GnaPWlTestsFixture<ov::opset11::SoftSign> test_instance({1, 128},
                                                            -10,
                                                            10,
                                                            ov::intel_gna::PWLApproximationMode::ACCURACY);
    test_instance.run();
}

TEST(GnaPwlTest, Log) {
    GnaPWlTestsFixture<ov::opset11::Log> test_instance({1, 128},
                                                       0.001,
                                                       2981,
                                                       ov::intel_gna::PWLApproximationMode::ACCURACY);
    test_instance.run();
}

TEST(GnaPwlTest, Power) {
    for (float exp = 1.0; exp <= 2.2; exp += 0.1) {
        GnaPWlTestsFixture<ov::opset11::Power> test_instance({1, 128},
                                                             AreFpEq(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
                                                             16,
                                                             exp,
                                                             ov::intel_gna::PWLApproximationMode::ACCURACY);
        test_instance.run();
    }
}

TEST(GnaPwlTest, Sigmoid_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Sigmoid> test_instance({1, 128},
                                                           -10.0,
                                                           10.0,
                                                           ov::intel_gna::PWLApproximationMode::PERFORMANCE);
    test_instance.run();
}

TEST(GnaPwlTest, Tanh_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Tanh> test_instance({1, 128},
                                                        -5.0,
                                                        5.0,
                                                        ov::intel_gna::PWLApproximationMode::PERFORMANCE);
    test_instance.run();
}

TEST(GnaPwlTest, Exp_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Exp> test_instance({1, 128},
                                                       -std::log2(INT16_MAX),
                                                       std::log10(INT16_MAX),
                                                       ov::intel_gna::PWLApproximationMode::PERFORMANCE);
    test_instance.run();
}

TEST(GnaPwlTest, SoftSign_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::SoftSign> test_instance({1, 128},
                                                            -10,
                                                            10,
                                                            ov::intel_gna::PWLApproximationMode::PERFORMANCE);
    test_instance.run();
}

TEST(GnaPwlTest, Log_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Log> test_instance({1, 128},
                                                       0.001,
                                                       2981,
                                                       ov::intel_gna::PWLApproximationMode::PERFORMANCE);
    test_instance.run();
}

TEST(GnaPwlTest, Power_PERFORMANCE) {
    for (float exp = 1.0; exp <= 2.2; exp += 0.1) {
        GnaPWlTestsFixture<ov::opset11::Power> test_instance({1, 128},
                                                             AreFpEq(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
                                                             16,
                                                             exp,
                                                             ov::intel_gna::PWLApproximationMode::PERFORMANCE);
        test_instance.run();
    }
}

TEST(GnaPwlTest, Sigmoid_fq) {
    GnaPWlTestsFixture<ov::opset11::Sigmoid> test_instance({1, 128},
                                                           -10.0,
                                                           10.0,
                                                           ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                           true);
    test_instance.run();
}

TEST(GnaPwlTest, Tanh_fq) {
    GnaPWlTestsFixture<ov::opset11::Tanh> test_instance({1, 128},
                                                        -5.0,
                                                        5.0,
                                                        ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                        true);
    test_instance.run();
}

TEST(GnaPwlTest, Exp_fq) {
    GnaPWlTestsFixture<ov::opset11::Exp> test_instance({1, 128},
                                                       -std::log2(INT16_MAX),
                                                       std::log10(INT16_MAX),
                                                       ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                       true);
    test_instance.run();
}

TEST(GnaPwlTest, SoftSign_fq) {
    GnaPWlTestsFixture<ov::opset11::SoftSign> test_instance({1, 128},
                                                            -10,
                                                            10,
                                                            ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                            true);
    test_instance.run();
}

TEST(GnaPwlTest, Log_fq) {
    GnaPWlTestsFixture<ov::opset11::Log> test_instance({1, 128},
                                                       0.001,
                                                       2981,
                                                       ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                       true);
    test_instance.run();
}

TEST(GnaPwlTest, Power_fq) {
    for (float exp = 1.0; exp <= 2.2; exp += 0.1) {
        GnaPWlTestsFixture<ov::opset11::Power> test_instance({1, 128},
                                                             AreFpEq(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
                                                             16,
                                                             exp,
                                                             ov::intel_gna::PWLApproximationMode::ACCURACY,
                                                             true);
        test_instance.run();
    }
}

TEST(GnaPwlTest, Sigmoid_fq_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Sigmoid> test_instance({1, 128},
                                                           -10.0,
                                                           10.0,
                                                           ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                           true);
    test_instance.run();
}

TEST(GnaPwlTest, Tanh_fq_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Tanh> test_instance({1, 128},
                                                        -5.0,
                                                        5.0,
                                                        ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                        true);
    test_instance.run();
}

TEST(GnaPwlTest, Exp_fq_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Exp> test_instance({1, 128},
                                                       -std::log2(INT16_MAX),
                                                       std::log10(INT16_MAX),
                                                       ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                       true);
    test_instance.run();
}

TEST(GnaPwlTest, SoftSign_fq_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::SoftSign> test_instance({1, 128},
                                                            -10,
                                                            10,
                                                            ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                            true);
    test_instance.run();
}

TEST(GnaPwlTest, Log_fq_PERFORMANCE) {
    GnaPWlTestsFixture<ov::opset11::Log> test_instance({1, 128},
                                                       0.001,
                                                       2981,
                                                       ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                       true);
    test_instance.run();
}

TEST(GnaPwlTest, Power_fq_PERFORMANCE) {
    for (float exp = 1.0; exp <= 2.2; exp += 0.1) {
        GnaPWlTestsFixture<ov::opset11::Power> test_instance({1, 128},
                                                             AreFpEq(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
                                                             16,
                                                             exp,
                                                             ov::intel_gna::PWLApproximationMode::PERFORMANCE,
                                                             true);
        test_instance.run();
    }
}
}  // namespace
