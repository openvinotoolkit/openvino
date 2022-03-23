// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/pwl_approximation.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <transformations/init_node_info.hpp>

namespace pwl_test {
template<typename T>
std::shared_ptr<ngraph::Function> CreateActivationFunction(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    auto f = std::make_shared<T>(input_params);
    auto result = std::make_shared<ngraph::opset8::Result>(f);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

template<typename T>
std::shared_ptr<ngraph::Function> CreateActivationFunction(const ngraph::Shape& input_shape, double exp) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, input_shape);
    auto exponents = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {exp});
    auto f = std::make_shared<T>(input_params, exponents);
    auto result = std::make_shared<ngraph::opset8::Result>(f);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}
} // namespace pwl_test

namespace {
void RunTest(const std::shared_ptr<ngraph::Function>& func,
             const std::shared_ptr<ngraph::Function>& reference_func,
             float lower_bound,
             float upper_bound) {
    double max_error_percent = 1;
    {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::PWLApproximation>(max_error_percent);
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    auto shape = func->input().get_node_shared_ptr()->get_output_shape(0);
    ov::runtime::TensorVector result(1);
    std::vector<float> data = CommonTestUtils::generate_float_numbers(ov::shape_size(shape), lower_bound, upper_bound);
    ov::runtime::Tensor input{ov::element::f32, shape, data.data()};
    ASSERT_TRUE(func->evaluate(result, ov::runtime::TensorVector{input}));

    ov::runtime::TensorVector result_ref(1);
    ASSERT_TRUE(reference_func->evaluate(result_ref, ov::runtime::TensorVector{input}));

    const float* result_data = result[0].data<float>();
    const float* result_ref_data = result_ref[0].data<float>();
    for (size_t i = 0; i < result[0].get_size(); i++) {
        double delta = std::abs(result_data[i] - result_ref_data[i]);
        ASSERT_TRUE(delta <= max_error_percent);
    }
}
} // namespace

TEST(GnaPwlTest, Sigmoid) {
    RunTest(
        pwl_test::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 100}),
        pwl_test::CreateActivationFunction<ngraph::opset8::Sigmoid>({1, 100}),
        -10,
        10);
}

TEST(GnaPwlTest, Tanh) {
    RunTest(
        pwl_test::CreateActivationFunction<ngraph::opset8::Tanh>({1, 32}),
        pwl_test::CreateActivationFunction<ngraph::opset8::Tanh>({1, 32}),
        -5,
        5);
}

TEST(GnaPwlTest, Exp) {
    RunTest(
        pwl_test::CreateActivationFunction<ngraph::opset8::Exp>({1, 32}),
        pwl_test::CreateActivationFunction<ngraph::opset8::Exp>({1, 32}),
        -std::log2(INT16_MAX),
        std::log10(INT16_MAX));
}

TEST(GnaPwlTest, SoftSign) {
    RunTest(
        pwl_test::CreateActivationFunction<ov::intel_gna::op::SoftSign>({1, 32}),
        pwl_test::CreateActivationFunction<ov::intel_gna::op::SoftSign>({1, 32}),
        -10,
        10);
}

TEST(GnaPwlTest, Log) {
    RunTest(
        pwl_test::CreateActivationFunction<ngraph::opset8::Log>({1, 32}),
        pwl_test::CreateActivationFunction<ngraph::opset8::Log>({1, 32}),
        0.001,
        2981);
}

TEST(GnaPwlTest, Power) {
    for (float exp = 1; exp <= 2.2; exp+=0.1) {
        RunTest(
            pwl_test::CreateActivationFunction<ngraph::opset8::Power>({1, 32}, exp),
            pwl_test::CreateActivationFunction<ngraph::opset8::Power>({1, 32}, exp),
            GNAPluginNS::details::are_floats_equal(std::fmod(exp, 1.0), 0.0) ? -16 : 0,
            16);
    }
}
