// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_mul_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <tuple>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;

namespace LayerTestsDefinitions {

using FQMulFusionParams = std::tuple<Shape,   // FQ data shape
                                     Shape,   // in_* shape
                                     Shape,   // out_* shape
                                     Shape,   // Mul constant shape
                                     Shape>;  // Expected shape of the new out_* constants

class FQMulFusion : public testing::WithParamInterface<FQMulFusionParams>, public ov::test::TestsCommon {
public:
    void SetUp() override {
        Shape data_shape, in_shape, out_shape, mul_const_shape, expected_out_shape;
        std::tie(data_shape, in_shape, out_shape, mul_const_shape, expected_out_shape) = this->GetParam();

        const auto data = opset4::Constant::create(element::Type_t::f32, data_shape, {0.0f});
        const auto in_low = opset4::Constant::create(element::Type_t::f32, in_shape, {-0.5f});
        const auto in_high = opset4::Constant::create(element::Type_t::f32, in_shape, {0.5f});
        const auto out_low = opset4::Constant::create(element::Type_t::f32, out_shape, {0.0f});
        const auto out_high = opset4::Constant::create(element::Type_t::f32, out_shape, {100.0f});
        const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 255);

        std::vector<float> mul_const(shape_size(mul_const_shape));
        std::iota(mul_const.begin(), mul_const.end(), 0.0f);
        const auto mul_value = opset4::Constant::create(element::Type_t::f32, mul_const_shape, mul_const);
        const auto mul = std::make_shared<opset4::Multiply>(fq, mul_value);

        m_model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{}, "FQMulFusion");

        const auto expected_data = opset4::Constant::create(element::Type_t::f32, data_shape, {0.0f});
        const auto expected_in_low = opset4::Constant::create(element::Type_t::f32, in_shape, {-0.5f});
        const auto expected_in_high = opset4::Constant::create(element::Type_t::f32, in_shape, {0.5f});
        const auto expected_out_low = opset4::Constant::create(element::Type_t::f32, expected_out_shape, {0.0f});
        const auto expected_out_high = opset4::Constant::create(element::Type_t::f32, expected_out_shape, {314.0f});

        const auto expected_fq = std::make_shared<opset4::FakeQuantize>(expected_data,
                                                                        expected_in_low,
                                                                        expected_in_high,
                                                                        expected_out_low,
                                                                        expected_out_high,
                                                                        255);

        m_expected_model =
            std::make_shared<ov::Model>(OutputVector{expected_fq}, ParameterVector{}, "FQMulFusion_expected");
    }

    std::shared_ptr<ov::Model> m_model;
    std::shared_ptr<ov::Model> m_expected_model;
};

TEST_P(FQMulFusion, ExpectFusion) {
    auto unh = std::make_shared<ov::pass::UniqueNamesHolder>();
    pass::Manager manager;
    manager.register_pass<ov::pass::InitUniqueNames>(unh);
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();
    manager.register_pass<ov::pass::CheckUniqueNames>(unh);

    manager.run_passes(m_model);
    OV_ASSERT_NO_THROW(check_rt_info(m_model));

    auto fc =
        FunctionsComparator::no_default().enable(FunctionsComparator::PRECISIONS).enable(FunctionsComparator::NODES);
    auto res = fc.compare(m_model, m_expected_model);
    ASSERT_TRUE(res.valid) << res.message;
};

namespace {
INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_4D_channel_0,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{64, 3, 7, 7}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{64, 1, 1, 1}),
                                            ::testing::Values(Shape{64, 1, 1, 1})));

INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_4D_channel_1,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{64, 3, 7, 7}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1, 3, 1, 1}),
                                            ::testing::Values(Shape{1, 3, 1, 1})));

INSTANTIATE_TEST_SUITE_P(ScalarFQParams_C6_scalar,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{64, 3, 7, 7}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{})));

INSTANTIATE_TEST_SUITE_P(FQOutputs1D_C6_scalar,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{64, 3, 7, 7}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_NHWC_C6_scalar,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 7, 7, 3}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1, 1, 1, 3}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1, 1, 1, 3})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_NCHW_C6_scalar,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 3, 7, 7}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1, 3, 1, 1}),
                                            ::testing::Values(Shape{}),
                                            ::testing::Values(Shape{1, 3, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_with_channel_dimension,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_per__multiplier_with_channel,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D_with_channel__multiplier_4D_per_tensor,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1})));

INSTANTIATE_TEST_SUITE_P(FQInputs_4D__multiplier_channel_3rd_dim,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1, 1, 3, 1}),
                                            ::testing::Values(Shape{1, 64, 3, 1})));

INSTANTIATE_TEST_SUITE_P(FQOutputs_1D__multiplier_3D,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 64, 1, 1}),
                                            ::testing::Values(Shape{1}),
                                            ::testing::Values(Shape{1, 3, 1}),
                                            ::testing::Values(Shape{1, 1, 3, 1})));

INSTANTIATE_TEST_SUITE_P(FQInOUt_ones__multiplier_4D_with_channel,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 1, 1, 1}),
                                            ::testing::Values(Shape{1, 64, 3, 3}),
                                            ::testing::Values(Shape{1, 64, 3, 3})));

INSTANTIATE_TEST_SUITE_P(FQInOUt_ones__multiplier_3D,
                         FQMulFusion,
                         ::testing::Combine(::testing::Values(Shape{1, 128, 512}),
                                            ::testing::Values(Shape{1}),
                                            ::testing::Values(Shape{1}),
                                            ::testing::Values(Shape{512}),
                                            ::testing::Values(Shape{1, 1, 512})));

TEST(FQMulFusion_NonConstInputs, AllInputsNonConst) {
    const auto data = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{1, 3, 224, 224});
    const auto in_low = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto in_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto out_low = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto out_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{}, {3.14f});
    const auto mul = std::make_shared<opset4::Multiply>(fq, mul_value);

    auto model =
        std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{data, in_low, in_high, out_low, out_high});

    const auto expected_out_low = std::make_shared<opset4::Multiply>(out_low, mul_value);
    const auto expected_out_high = std::make_shared<opset4::Multiply>(out_high, mul_value);

    const auto expected_fq =
        std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function =
        std::make_shared<ov::Model>(OutputVector{expected_fq},
                                    ParameterVector{data, in_low, in_high, out_low, out_high});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto res = compare_functions(model, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_NonConstInputs, FQ_out_high_const) {
    const auto data = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{1, 3, 224, 224});
    const auto in_low = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto in_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto out_low = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto out_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {100.0f});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{}, {3.14f});
    const auto mul = std::make_shared<opset4::Multiply>(fq, mul_value);

    auto model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{data, in_low, in_high, out_low});

    const auto expected_out_low = std::make_shared<opset4::Multiply>(out_low, mul_value);
    // this constant should be created by constant folding of the last FQ input
    const auto expected_out_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {314.0f});

    const auto expected_fq =
        std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function =
        std::make_shared<ov::Model>(OutputVector{expected_fq}, ParameterVector{data, in_low, in_high, out_low});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto res = compare_functions(model, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_FQ_Mul_inputs, FQ_out_to_mul_input_2) {
    const auto data = opset4::Constant::create(element::Type_t::f32, Shape{1, 3, 224, 224}, {0.0f});
    const auto in_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {-0.5f});
    const auto in_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.5f});
    const auto out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.0f});
    const auto out_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {100.0f});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{}, {3.14f});
    // here the FQ's output is passed to the second input of the Mul operation
    const auto mul = std::make_shared<opset4::Multiply>(mul_value, fq);

    auto model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{});

    const auto expected_out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.0f});
    const auto expected_out_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {314.0f});

    const auto expected_fq =
        std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function = std::make_shared<ov::Model>(OutputVector{expected_fq}, ParameterVector{});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto res = compare_functions(model, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(FQMulFusion_FQ_Mul_inputs, FQ_out_to_mul_input_2_param) {
    const auto data = opset4::Constant::create(element::Type_t::f32, Shape{1, 3, 224, 224}, {0.0f});
    const auto in_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {-0.5f});
    const auto in_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.5f});
    const auto out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.0f});
    // out_high is a parameter, which means it should not be constant folded
    const auto out_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{}, {3.14f});
    // and here the output of FQ is passed as the second input of Mul
    const auto mul = std::make_shared<opset4::Multiply>(mul_value, fq);

    auto model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{out_high});

    const auto expected_out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.0f});
    const auto expected_out_high = std::make_shared<opset4::Multiply>(out_high, mul_value);

    const auto expected_fq =
        std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function = std::make_shared<ov::Model>(OutputVector{expected_fq}, ParameterVector{out_high});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto res = compare_functions(model, expected_function);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, FakeQuantizeMultiplyFusionNegative) {
    const auto data = opset4::Constant::create(element::Type_t::f32, Shape{1, 300, 1}, {0.0f});
    const auto in_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {-0.5f});
    const auto in_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.5f});
    const auto out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {0.0f});
    // out_high is a parameter, which means it should not be constant folded
    const auto out_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{1, 300, 16}, {3.14f});
    // and here the output of FQ is passed as the second input of Mul
    const auto mul = std::make_shared<opset4::Multiply>(mul_value, fq);

    auto model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{out_high});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    ASSERT_EQ(model->get_output_shape(0), Shape({1, 300, 16}));
}

TEST(TransformationTests, FakeQuantizeMultiplyFusionMulConstWithEqualValues) {
    const auto data = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{1, 3, 224, 224});
    const auto in_low = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto in_high = std::make_shared<opset4::Parameter>(element::Type_t::f32, Shape{});
    const auto out_low = opset4::Constant::create(element::Type_t::f32, Shape{}, {1.0f});
    const auto out_high = opset4::Constant::create(element::Type_t::f32, Shape{}, {100.0f});
    const auto fq = std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, out_low, out_high, 42);

    const auto mul_value = opset4::Constant::create(element::Type_t::f32, Shape{1, 3, 1, 1}, {3, 3, 3});
    const auto mul = std::make_shared<opset4::Multiply>(fq, mul_value);

    auto model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{data, in_low, in_high});

    const auto expected_out_low = opset4::Constant::create(element::Type_t::f32, Shape{1}, {3.0f});
    // this constant should be created by constant folding of the last FQ input
    const auto expected_out_high = opset4::Constant::create(element::Type_t::f32, Shape{1}, {300.0f});

    const auto expected_fq =
        std::make_shared<opset4::FakeQuantize>(data, in_low, in_high, expected_out_low, expected_out_high, 42);

    const auto expected_function =
        std::make_shared<ov::Model>(OutputVector{expected_fq}, ParameterVector{data, in_low, in_high});

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::FakeQuantizeMulFusion>();

    manager.run_passes(model);
    OV_ASSERT_NO_THROW(check_rt_info(model));

    const auto res = compare_functions(model, expected_function, true);
    ASSERT_TRUE(res.first) << res.second;
}

}  // namespace

}  // namespace LayerTestsDefinitions
