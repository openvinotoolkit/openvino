// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/common_optimizations/transpose_to_pwl.hpp"

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/serialize.hpp>
#include <transformations/init_node_info.hpp>
#include <frontend_manager/frontend_manager.hpp>

namespace pwl_test {
std::shared_ptr<ngraph::Function> CreateSigmoid(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
    auto sigmoid = std::make_shared<ngraph::opset1::Sigmoid>(input_params);
    auto result = std::make_shared<ngraph::opset1::Result>(sigmoid);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> CreateTanh(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
    auto tanh = std::make_shared<ngraph::opset1::Tanh>(input_params);
    auto result = std::make_shared<ngraph::opset1::Result>(tanh);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> CreateExp(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
    auto exp = std::make_shared<ngraph::opset1::Exp>(input_params);
    auto result = std::make_shared<ngraph::opset1::Result>(exp);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> CreateAbs(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
    auto abs = std::make_shared<ngraph::opset1::Abs>(input_params);
    auto result = std::make_shared<ngraph::opset1::Result>(abs);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

std::shared_ptr<ngraph::Function> CreateSign(const ngraph::Shape& input_shape) {
    auto input_params = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
    auto sign = std::make_shared<ngraph::opset1::Sign>(input_params);
    auto result = std::make_shared<ngraph::opset1::Result>(sign);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}
} // namespace pwl_test

namespace {
void RunTest(const std::shared_ptr<ngraph::Function>& func,
             const std::shared_ptr<ngraph::Function>& reference_func,
             float lower_bound,
             float upper_bound) {
    {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::TransposeToPwl>();
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
        std::cout << "delta: " << delta << " " << result_data[i] << " " << result_ref_data[i] << '\n';
        ASSERT_TRUE(delta <= 0.005);
    }
}
} // namespace

TEST(PwlTest, Sigmoid) {
    RunTest(
        pwl_test::CreateSigmoid({1, 100}),
        pwl_test::CreateSigmoid({1, 100}),
        -10,
        10);
}

TEST(PwlTest, Tanh) {
    RunTest(
        pwl_test::CreateTanh({1, 32}),
        pwl_test::CreateTanh({1, 32}),
        -5,
        5);
}

/*TEST(PwlTest, Exp) {
    RunTest(
        pwl_test::CreateExp({1, 32}),
        pwl_test::CreateExp({1, 32}),
        0,
        log(INT16_MAX));
}*/

TEST(PwlTest, Abs) {
    RunTest(
        pwl_test::CreateAbs({1, 32}),
        pwl_test::CreateAbs({1, 32}),
        -1,
        1);
}

TEST(PwlTest, Sign) {
    RunTest(
        pwl_test::CreateSign({1, 32}),
        pwl_test::CreateSign({1, 32}),
        -1,
        1);
}
