// Copyright (C) 2018-2021 Intel Corporation
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
#include <transformations/init_node_info.hpp>

namespace LayerTestsDefinitions {

using FQMulFusionParams =
    std::tuple<ngraph::Shape,  // FQ data shape
               ngraph::Shape,  // in_* shape
               ngraph::Shape,  // out_* shape
               ngraph::Shape,  // Mul constant shape
               ngraph::Shape>; // Expected shape of the new out_* constants

class FQMulFusion : public testing::WithParamInterface<FQMulFusionParams>,
                    public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        ngraph::Shape data_shape, in_shape, out_shape, mul_const_shape, expected_out_shape;
        std::tie(data_shape, in_shape, out_shape, mul_const_shape, expected_out_shape) =
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
            ngraph::element::Type_t::f32, expected_out_shape, {0.0f});
        const auto expected_out_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, expected_out_shape, {314.0f});

        const auto expected_fq =
            std::make_shared<ngraph::opset4::FakeQuantize>(expected_data,
                expected_in_low, expected_in_high, expected_out_low, expected_out_high, 255);

        m_expected_function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{expected_fq}, ngraph::ParameterVector{}, "FQMulFusion_expected");
    }

  std::shared_ptr<ngraph::Function> m_function;
  std::shared_ptr<ngraph::Function> m_expected_function;
};

TEST_P(FQMulFusion, ExpectFusion) {
  ngraph::pass::Manager manager;
  manager.register_pass<ngraph::pass::InitNodeInfo>();
  manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

  manager.run_passes(m_function);
  ASSERT_NO_THROW(check_rt_info(m_function));

  const auto res = compare_functions(m_function, m_expected_function);
  ASSERT_TRUE(res.first) << res.second;
};

namespace {
INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_4D_channel_0, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_4D_channel_1, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1, 1})));

INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_scalar, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{})));

INSTANTIATE_TEST_SUITE_P(FQOutputs1D_C6_scalar, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{64, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_NHWC_C6_scalar, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 7, 7, 3}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 3}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 3})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_NCHW_C6_scalar, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 3, 7, 7}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                           ::testing::Values(ngraph::Shape{}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_with_channel_dimension, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_per__multiplier_with_channel, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_with_channel__multiplier_4D_per_tensor, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D__multiplier_channel_3rd_dim, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 1, 3, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 3, 1})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_1D__multiplier_3D, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 64, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1}),
                                           ::testing::Values(ngraph::Shape{1, 3, 1})));

INSTANTIATE_TEST_SUITE_P(FQInOUt_ones__multiplier_4D_with_channel, FQMulFusion,
                        ::testing::Combine(::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                           ::testing::Values(ngraph::Shape{1, 64, 3, 3}),
                                           ::testing::Values(ngraph::Shape{1, 64, 3, 3})));

TEST(FQMulFusion_NonConstInputs, AllInputsNonConst) {
    const auto data = std::make_shared<ngraph::opset4::Parameter>(
        ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 224, 224});
    const auto in_low =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto in_high =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto out_low =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto out_high =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {3.14f});
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(fq, mul_value);

    auto function = std::make_shared<ngraph::Function>(ngraph::OutputVector{mul},
        ngraph::ParameterVector{data, in_low, in_high, out_low, out_high});

    const auto expected_out_low = std::make_shared<ngraph::opset4::Multiply>(out_low, mul_value);
    const auto expected_out_high = std::make_shared<ngraph::opset4::Multiply>(out_high, mul_value);

    const auto expected_fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function =
        std::make_shared<ngraph::Function>(ngraph::OutputVector{expected_fq},
            ngraph::ParameterVector{data, in_low, in_high, out_low, out_high});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const auto res = compare_functions(function, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_NonConstInputs, FQ_out_high_const) {
    const auto data = std::make_shared<ngraph::opset4::Parameter>(
        ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 224, 224});
    const auto in_low =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto in_high =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto out_low =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto out_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {100.0f});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {3.14f});
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(fq, mul_value);

    auto function = std::make_shared<ngraph::Function>(ngraph::OutputVector{mul},
        ngraph::ParameterVector{data, in_low, in_high, out_low});

    const auto expected_out_low = std::make_shared<ngraph::opset4::Multiply>(out_low, mul_value);
    // this constant should be created by constant folding of the last FQ input
    const auto expected_out_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {314.0f});

    const auto expected_fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function =
        std::make_shared<ngraph::Function>(ngraph::OutputVector{expected_fq},
            ngraph::ParameterVector{data, in_low, in_high, out_low});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const auto res = compare_functions(function, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_FQ_Mul_inputs, FQ_out_to_mul_input_2) {
    const auto data = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 224, 224}, {0.0f});
    const auto in_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {-0.5f});
    const auto in_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.5f});
    const auto out_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    const auto out_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {100.0f});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {3.14f});
    // here the FQ's output is passed to the second input of the Mul operation
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(mul_value, fq);

    auto function =
        std::make_shared<ngraph::Function>(ngraph::OutputVector{mul}, ngraph::ParameterVector{});

    const auto expected_out_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    const auto expected_out_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {314.0f});

    const auto expected_fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{expected_fq}, ngraph::ParameterVector{});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const auto res = compare_functions(function, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_FQ_Mul_inputs, FQ_out_to_mul_input_2_param) {
    const auto data = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 224, 224}, {0.0f});
    const auto in_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {-0.5f});
    const auto in_high = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.5f});
    const auto out_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    // out_high is a parameter, which means it should not be constant folded
    const auto out_high =
        std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {3.14f});
    // and here the output of FQ is passed as the second input of Mul
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(mul_value, fq);

    auto function = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{mul}, ngraph::ParameterVector{out_high});

    const auto expected_out_low = ngraph::opset4::Constant::create(
        ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    const auto expected_out_high = std::make_shared<ngraph::opset4::Multiply>(out_high, mul_value);

    const auto expected_fq = std::make_shared<ngraph::opset4::FakeQuantize>(
        data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function = std::make_shared<ngraph::Function>(
        ngraph::OutputVector{expected_fq}, ngraph::ParameterVector{out_high});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    const auto res = compare_functions(function, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FakeQuantizeMultiplyFusionNegative) {
    const auto data = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, ngraph::Shape{1, 300, 1}, {0.0f});
    const auto in_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, ngraph::Shape{}, {-0.5f});
    const auto in_high = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, ngraph::Shape{}, {0.5f});
    const auto out_low = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, ngraph::Shape{}, {0.0f});
    // out_high is a parameter, which means it should not be constant folded
    const auto out_high =
            std::make_shared<ngraph::opset4::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{});
    const auto fq = std::make_shared<ngraph::opset4::FakeQuantize>(
            data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = ngraph::opset4::Constant::create(
            ngraph::element::Type_t::f32, ngraph::Shape{1, 300, 16}, {3.14f});
    // and here the output of FQ is passed as the second input of Mul
    const auto mul = std::make_shared<ngraph::opset4::Multiply>(mul_value, fq);

    auto function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{mul}, ngraph::ParameterVector{out_high});

    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::FakeQuantizeMulFusion>();

    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    ASSERT_EQ(function->get_output_shape(0), ngraph::Shape({1, 300, 16}));
}


} // namespace

} // namespace LayerTestsDefinitions
