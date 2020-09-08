// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <memory>
#include <tuple>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/plugin_cache.hpp"

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/fq_mul_fusion.hpp>

namespace LayerTestsDefinitions {

using FQMulFusionParams =
    std::tuple<ngraph::Shape,  // FQ data shape
               ngraph::Shape,  // in_* shape
               ngraph::Shape,  // out_* shape
               ngraph::Shape>; // Mul constant shape

class FQMulFusion : public testing::WithParamInterface<FQMulFusionParams>,
                    public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        ngraph::Shape data_shape, in_shape, out_shape, mul_const_shape;
        std::tie(data_shape, in_shape, out_shape, mul_const_shape) =
            this->GetParam();

        const auto data = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, data_shape, {0.0f});
        const auto in_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, in_shape, {-0.5f});
        const auto in_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, in_shape, {0.5f});
        const auto out_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, out_shape, {0.0f});
        const auto out_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, out_shape, {100.0f});
        const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
            data, in_low, in_high, out_low, out_high, 255);

        const auto mul_value = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, mul_const_shape, {3.14f});
        const auto mul = std::make_shared<ngraph::opset4::Multiply>(fq, mul_value);

        m_function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{mul}, ngraph::ParameterVector{}, "FQMulFusion");

        const auto expected_data = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, data_shape, {0.0f});
        const auto expected_in_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, in_shape, {-0.5f});
        const auto expected_in_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, in_shape, {0.5f});
        const auto expected_out_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, out_shape, {0.0f});
        const auto expected_out_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, out_shape, {100.0f});

        const auto expected_mul_value = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, mul_const_shape, {3.14f});
        const auto expected_mul_out_low =
            std::make_shared<ngraph::opset4::Multiply>(expected_out_low, expected_mul_value);
        const auto expected_mul_out_high =
            std::make_shared<ngraph::opset4::Multiply>(expected_out_high, expected_mul_value);
        const auto expected_fq = std::make_shared<ngraph::opset4::FakeQuantize>(
            expected_data, expected_in_low, expected_in_high, expected_mul_out_low,
            expected_mul_out_high, 255);

        m_expected_function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{expected_fq}, ngraph::ParameterVector{}, "FQMulFusion_expected");
    }

  std::shared_ptr<ngraph::Function> m_function;
  std::shared_ptr<ngraph::Function> m_expected_function;
};

TEST_P(FQMulFusion, ExpectFusion) {
  ngraph::pass::Manager manager;
  manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

  manager.run_passes(m_function);

  const auto res = compare_functions(m_function, m_expected_function);
  ASSERT_TRUE(res.first) << res.second;
};

namespace {
INSTANTIATE_TEST_CASE_P(MulConstBcastedToDataShape, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_CASE_P(FQInputs1D, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_CASE_P(FQOutputs1D, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_CASE_P(FQInputsOutputs1D, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_CASE_P(MulConstScalar, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{})));
} // namespace

} // namespace LayerTestsDefinitions
