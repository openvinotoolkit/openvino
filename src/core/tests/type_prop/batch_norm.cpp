// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"

using namespace std;

struct BatchNormInferInputs {
    ov::element::Type in_et;
    ov::PartialShape in_shape;
    std::string in_name;
};

struct BatchNormInferParams {
    ov::element::Type data_batch_et;
    ov::PartialShape data_batch_ps;
    std::vector<BatchNormInferInputs> inputs;
    double epsilon;
};

template <class T>
std::shared_ptr<ov::Node> makeBatchNormOp(const BatchNormInferParams& p) {
    if (p.inputs.size() != 4) {
        throw runtime_error("BatchNormInference requires 4 additional inputs for batch"
                            "normalization transformation");
    }
    auto data_batch = make_shared<ov::op::v0::Parameter>(p.data_batch_et, p.data_batch_ps);
    auto gamma = make_shared<ov::op::v0::Parameter>(p.inputs[0].in_et, p.inputs[0].in_shape);
    auto beta = make_shared<ov::op::v0::Parameter>(p.inputs[1].in_et, p.inputs[1].in_shape);
    auto mean = make_shared<ov::op::v0::Parameter>(p.inputs[2].in_et, p.inputs[2].in_shape);
    auto variance = make_shared<ov::op::v0::Parameter>(p.inputs[3].in_et, p.inputs[3].in_shape);
    return make_shared<T>(data_batch, gamma, beta, mean, variance, p.epsilon);
}

template <class T>
class BatchNormTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(BatchNormTest);

TYPED_TEST_P(BatchNormTest, batch_norm_inference_basic_data_batch_rank_2) {
    ov::PartialShape data_batch_shape{10, 100};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape{100}, "gamma"},
                                                   {inputs_et, ov::PartialShape{100}, "beta"},
                                                   {inputs_et, ov::PartialShape{100}, "mean"},
                                                   {inputs_et, ov::PartialShape{100}, "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), inputs_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(data_batch_shape));
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_basic_data_batch_rank_4) {
    ov::PartialShape data_batch_shape{1, 10, 224, 224};
    ov::element::Type inputs_et = ov::element::f16;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape{10}, "gamma"},
                                                   {inputs_et, ov::PartialShape{10}, "beta"},
                                                   {inputs_et, ov::PartialShape{10}, "mean"},
                                                   {inputs_et, ov::PartialShape{10}, "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), inputs_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_static());
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(data_batch_shape));
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_inputs_rank_dynamic) {
    ov::PartialShape data_batch_shape{ov::PartialShape::dynamic()};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape::dynamic(), "gamma"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "beta"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "mean"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), inputs_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_data_batch_rank_static_channel_inputs_rank_dynamic) {
    ov::PartialShape data_batch_shape{64, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape::dynamic(), "gamma"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "beta"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "mean"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), inputs_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(
        ov::PartialShape{64, ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()}));
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_data_batch_rank_dynamic_some_channel_inputs_rank_static) {
    ov::PartialShape data_batch_shape{ov::PartialShape::dynamic()};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {{input_et, ov::PartialShape{ov::Dimension::dynamic()}, "gamma"},
                                                {input_et, ov::PartialShape::dynamic(), "beta"},
                                                {input_et, ov::PartialShape{ov::Dimension::dynamic()}, "mean"},
                                                {input_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), input_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).rank().is_dynamic());
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_data_batch_rank_static_some_channel_inputs_rank_static) {
    ov::PartialShape data_batch_shape{64, ov::Dimension::dynamic(), ov::Dimension::dynamic(), 224};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {{input_et, ov::PartialShape{3}, "gamma"},
                                                {input_et, ov::PartialShape::dynamic(), "beta"},
                                                {input_et, ov::PartialShape{3}, "mean"},
                                                {input_et, ov::PartialShape{ov::Dimension::dynamic()}, "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    auto bn = makeBatchNormOp<TypeParam>(params);

    ASSERT_EQ(bn->get_output_size(), 1);
    ASSERT_EQ(bn->get_output_element_type(0), input_et);
    ASSERT_TRUE(bn->get_output_partial_shape(0).same_scheme(ov::PartialShape{64, 3, ov::Dimension::dynamic(), 224}));
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_invalid_inputs_element_types) {
    ov::PartialShape data_batch_shape{10, 100};

    const std::vector<ov::element::Type> inputs_et{ov::element::i32, ov::element::u32, ov::element::boolean};

    double eps = 0.001;

    std::vector<BatchNormInferParams> bn_tests;
    for (const auto& et : inputs_et) {
        std::vector<BatchNormInferInputs> ch_inputs = {{et, ov::PartialShape{100}, "gamma"},
                                                       {et, ov::PartialShape{100}, "beta"},
                                                       {et, ov::PartialShape{100}, "mean"},
                                                       {et, ov::PartialShape{100}, "variance"}};

        bn_tests.push_back(BatchNormInferParams{et, data_batch_shape, ch_inputs, eps});
    }

    for (const auto& params : bn_tests) {
        try {
            auto bn = makeBatchNormOp<TypeParam>(params);
            FAIL() << "Invalid input element types not detected";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "Input element types must be floating-point");
        } catch (...) {
            FAIL() << "Input element types check failed for unexpected reason";
        }
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_incompatible_inputs_element_types) {
    // Data batch input element type and shape
    const ov::element::Type data_batch_et = ov::element::f32;
    const ov::PartialShape data_batch_ps{10, 200};

    // Invalid combination of element types of gamma/beta/mean/variance inputs
    vector<BatchNormInferInputs> bn_ch_inputs = {{ov::element::f32, ov::PartialShape{200}, "gamma"},
                                                 {ov::element::f32, ov::PartialShape{200}, "beta"},
                                                 {ov::element::f32, ov::PartialShape{200}, "mean"},
                                                 {ov::element::f32, ov::PartialShape{200}, "variance"}};

    const double epsilon = 0.001;

    std::vector<BatchNormInferParams> bn_params;
    bn_params.push_back(BatchNormInferParams{ov::element::f16, data_batch_ps, bn_ch_inputs, epsilon});

    for (size_t i = 0; i < bn_ch_inputs.size(); i++) {
        std::vector<BatchNormInferInputs> inputs = bn_ch_inputs;
        (inputs[i]).in_et = ov::element::f16;
        bn_params.push_back(BatchNormInferParams{data_batch_et, data_batch_ps, inputs, epsilon});
    }

    // Run tests with incompatible input element types
    for (const auto& bn_p : bn_params) {
        try {
            auto bn = makeBatchNormOp<TypeParam>(bn_p);
            FAIL() << "Incompatible input element types not detected";
        } catch (const ov::NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "Input element types do not match");
        } catch (...) {
            FAIL() << "Input element types check failed for unexpected reason";
        }
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_invalid_data_batch_input_rank) {
    ov::PartialShape data_batch_shape{ov::Dimension::dynamic()};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape::dynamic(), "gamma"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "beta"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "mean"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Data batch input with invalid rank 1 not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Input argument must have rank of at least 2 (input argument shape: [?])");
    } catch (...) {
        FAIL() << "Data batch input rank check failed for unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_incompatible_channel_input_ranks) {
    ov::PartialShape data_batch_shape{ov::PartialShape::dynamic()};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {{input_et, ov::PartialShape{3, ov::Dimension::dynamic()}, "gamma"},
                                                {input_et, ov::PartialShape::dynamic(), "beta"},
                                                {input_et, ov::PartialShape{ov::Dimension::dynamic()}, "mean"},
                                                {input_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Incompatible gamma/beta/mean/variance input ranks not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Shapes for gamma/beta/mean/variance do not match");
    } catch (...) {
        FAIL() << "gamma/beta/mean/variance input ranks check failed for unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_incompatible_channel_inputs_channel_count) {
    ov::PartialShape data_batch_shape{ov::PartialShape::dynamic()};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {{input_et, ov::PartialShape{3}, "gamma"},
                                                {input_et, ov::PartialShape::dynamic(), "beta"},
                                                {input_et, ov::PartialShape{4}, "mean"},
                                                {input_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Incompatible gamma/beta/mean/variance inputs channel count not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Shapes for gamma/beta/mean/variance do not match");
    } catch (...) {
        FAIL() << "gamma/beta/mean/variance inputs channel count check failed for unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_invalid_channel_inputs_rank) {
    ov::PartialShape data_batch_shape{ov::PartialShape::dynamic()};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {
        {input_et, ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, "gamma"},
        {input_et, ov::PartialShape::dynamic(), "beta"},
        {input_et, ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic()}, "mean"},
        {input_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Invalid rank of gamma/beta/mean/variance inputs not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Shape for gamma/beta/mean/variance ([?,?]) does not have rank 1");
    } catch (...) {
        FAIL() << "gamma/beta/mean/variance inputs rank check failed for unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_incompatible_data_batch_and_channel_inputs_channel_count) {
    ov::PartialShape data_batch_shape{64, 4, ov::Dimension::dynamic(), 224};
    ov::element::Type input_et = ov::element::f32;

    std::vector<BatchNormInferInputs> inputs = {{input_et, ov::PartialShape{3}, "gamma"},
                                                {input_et, ov::PartialShape::dynamic(), "beta"},
                                                {input_et, ov::PartialShape{3}, "mean"},
                                                {input_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{input_et, data_batch_shape, inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Incompatible data batch and gamma/beta/mean/variance channel count not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             "Input channel dimension (4) does not match "
                             "shape for gamma/beta/mean/variance ([3])");
    } catch (...) {
        FAIL() << "Data batch and gamma/beta/mean/variance channel count check failed for "
                  "unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_invalid_input_channels_count_zero) {
    ov::PartialShape data_batch_shape{ov::Dimension::dynamic(), 0, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape::dynamic(), "gamma"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "beta"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "mean"},
                                                   {inputs_et, ov::PartialShape::dynamic(), "variance"}};

    double epsilon = 0.001;

    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, epsilon};
    try {
        auto bn = makeBatchNormOp<TypeParam>(params);
        FAIL() << "Data batch channel count zero not detected";
    } catch (const ov::NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), "Channel count must be at least 1");
    } catch (...) {
        FAIL() << "Data batch channel count check failed for unexpected reason";
    }
}

TYPED_TEST_P(BatchNormTest, batch_norm_inference_invalid_epsilon) {
    ov::PartialShape data_batch_shape{10, 100};
    ov::element::Type inputs_et = ov::element::f32;

    std::vector<BatchNormInferInputs> ch_inputs = {{inputs_et, ov::PartialShape{100}, "gamma"},
                                                   {inputs_et, ov::PartialShape{100}, "beta"},
                                                   {inputs_et, ov::PartialShape{100}, "mean"},
                                                   {inputs_et, ov::PartialShape{100}, "variance"}};

    double eps_neg = -1.0;
    const BatchNormInferParams params{inputs_et, data_batch_shape, ch_inputs, eps_neg};
    OV_EXPECT_THROW_HAS_SUBSTRING(std::ignore = makeBatchNormOp<TypeParam>(params),
                                  ov::NodeValidationFailure,
                                  "Attribute 'epsilon' must be non negative value");
}

REGISTER_TYPED_TEST_SUITE_P(BatchNormTest,
                            batch_norm_inference_basic_data_batch_rank_2,
                            batch_norm_inference_basic_data_batch_rank_4,
                            batch_norm_inference_inputs_rank_dynamic,
                            batch_norm_inference_data_batch_rank_static_channel_inputs_rank_dynamic,
                            batch_norm_inference_data_batch_rank_dynamic_some_channel_inputs_rank_static,
                            batch_norm_inference_data_batch_rank_static_some_channel_inputs_rank_static,
                            batch_norm_inference_invalid_inputs_element_types,
                            batch_norm_inference_incompatible_inputs_element_types,
                            batch_norm_inference_invalid_data_batch_input_rank,
                            batch_norm_inference_incompatible_channel_input_ranks,
                            batch_norm_inference_incompatible_channel_inputs_channel_count,
                            batch_norm_inference_invalid_channel_inputs_rank,
                            batch_norm_inference_incompatible_data_batch_and_channel_inputs_channel_count,
                            batch_norm_inference_invalid_input_channels_count_zero,
                            batch_norm_inference_invalid_epsilon);

using Types = ::testing::Types<ov::op::v0::BatchNormInference, ov::op::v5::BatchNormInference>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, BatchNormTest, Types);
