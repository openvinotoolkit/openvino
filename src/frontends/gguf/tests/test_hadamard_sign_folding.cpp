// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <limits>

#include "gtest/gtest.h"
#include "hadamard_sign_folding.hpp"
#include "op_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

namespace {
using Constant = ov::op::v0::Constant;
using Multiply = ov::op::v1::Multiply;
using ov::frontend::gguf::fold_hadamard_input_signs;

std::shared_ptr<Multiply> marked_signs(const ov::Output<ov::Node>& data,
                                       const ov::Output<ov::Node>& signs,
                                       bool reversed = false) {
    auto result = reversed ? std::make_shared<Multiply>(signs, data) : std::make_shared<Multiply>(data, signs);
    result->get_rt_info()["gguf_hadamard_input_sign"] = true;
    return result;
}

ov::Tensor input_tensor(const ov::element::Type& type) {
    const std::vector<float> values{1.125f, -2.5f, 3.75f, -4.25f, -5.5f, 6.125f, -7.25f, 8.5f};
    return type == ov::element::f16 ? ov_gguf_test::make_f16_tensor({2, 4}, values)
                                    : ov_gguf_test::make_f32_tensor({2, 4}, values);
}

ov::TensorVector evaluate(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
    ov::TensorVector outputs;
    for (const auto& output : model->outputs())
        outputs.emplace_back(output.get_element_type(), output.get_shape());
    EXPECT_TRUE(model->evaluate(outputs, {input}));
    return outputs;
}

void expect_identical(const ov::Tensor& actual, const ov::Tensor& expected) {
    ASSERT_EQ(actual.get_element_type(), expected.get_element_type());
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.get_byte_size()), 0);
}

void expect_identical(const ov::TensorVector& actual, const ov::TensorVector& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        SCOPED_TRACE(i);
        expect_identical(actual[i], expected[i]);
    }
}

class GGUFHadamardSignFolding : public ::testing::TestWithParam<ov::element::Type> {};

TEST_P(GGUFHadamardSignFolding, PreservesNumericsSignedZeroSharedConstantsAndMetadata) {
    const auto type = GetParam();
    const auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2, 4});
    input->set_friendly_name("x");
    input->output(0).set_names({"x"});
    const auto scale = Constant::create(type, {4}, {0.75f, -0.375f, 0.0f, -0.0f});
    const auto signs = Constant::create(type, {4}, {-1, 1, -1, -1});
    const auto scaled = std::make_shared<Multiply>(scale, input);
    const auto rotation = marked_signs(scaled, signs, true);
    scale->get_rt_info()["scale_constant"] = std::string("scale");
    signs->get_rt_info()["sign_constant"] = std::string("signs");
    scaled->get_rt_info()["scale_metadata"] = 42;
    rotation->get_rt_info()["rotation_metadata"] = 7;
    scaled->get_rt_info()["common_metadata"] = 11;
    rotation->get_rt_info()["common_metadata"] = 11;
    scaled->output(0).get_rt_info()["scale_output_metadata"] = 5;
    rotation->output(0).get_rt_info()["rotation_output_metadata"] = 6;
    rotation->output(0).set_names({"rotated"});
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{rotation, scale}, ov::ParameterVector{input});
    const auto values = input_tensor(type);
    const auto expected = evaluate(model, values);
    const auto cpu_expected = ov_gguf_test::run_on_cpu(model, {{"x", values}});

    fold_hadamard_input_signs(model);

    const auto replacement = ov::as_type_ptr<Multiply>(model->get_results()[0]->input_value(0).get_node_shared_ptr());
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->input_value(0), input->output(0));
    const auto folded = ov::as_type_ptr<Constant>(replacement->input_value(1).get_node_shared_ptr());
    ASSERT_NE(folded, nullptr);
    EXPECT_EQ(folded->get_element_type(), type);
    const auto folded_reference = Constant::create(type, {4}, {-0.75f, -0.375f, -0.0f, 0.0f});
    expect_identical(folded->get_tensor_view(), folded_reference->get_tensor_view());
    EXPECT_EQ(model->get_results()[1]->input_value(0), scale->output(0));
    EXPECT_EQ(replacement->get_rt_info().at("scale_metadata").as<int>(), 42);
    EXPECT_EQ(replacement->get_rt_info().at("rotation_metadata").as<int>(), 7);
    EXPECT_EQ(replacement->get_rt_info().at("common_metadata").as<int>(), 11);
    EXPECT_EQ(replacement->get_rt_info().count("gguf_hadamard_input_sign"), 0u);
    EXPECT_EQ(folded->get_rt_info().at("scale_constant").as<std::string>(), "scale");
    EXPECT_EQ(folded->get_rt_info().at("sign_constant").as<std::string>(), "signs");
    EXPECT_EQ(replacement->output(0).get_rt_info().at("scale_output_metadata").as<int>(), 5);
    EXPECT_EQ(replacement->output(0).get_rt_info().at("rotation_output_metadata").as<int>(), 6);
    EXPECT_EQ(replacement->output(0).get_names().count("rotated"), 1u);
    expect_identical(evaluate(model, values), expected);
    expect_identical(ov_gguf_test::run_on_cpu(model, {{"x", values}}), cpu_expected);
}

TEST_P(GGUFHadamardSignFolding, IdenticalSignConsumersAndShapeConsumersShareFoldedScale) {
    const auto type = GetParam();
    const auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2, 4});
    const auto scale = Constant::create(type, {}, {0.375f});
    const auto scaled = std::make_shared<Multiply>(input, scale);
    const auto first = marked_signs(scaled, Constant::create(type, {4}, {-1, 1, -1, 1}));
    const auto second = marked_signs(scaled, Constant::create(type, {4}, {-1, 1, -1, 1}), true);
    const auto shape = std::make_shared<ov::op::v3::ShapeOf>(scaled);
    const auto old_shape = std::make_shared<ov::op::v0::ShapeOf>(scaled);
    const auto model =
        std::make_shared<ov::Model>(ov::OutputVector{first, second, shape, old_shape}, ov::ParameterVector{input});
    const auto values = input_tensor(type);
    const auto expected = evaluate(model, values);

    fold_hadamard_input_signs(model);

    const auto replacement = model->get_results()[0]->input_value(0);
    ASSERT_TRUE(ov::is_type<Multiply>(replacement.get_node()));
    EXPECT_EQ(model->get_results()[1]->input_value(0), replacement);
    EXPECT_EQ(shape->input_value(0), replacement);
    EXPECT_EQ(old_shape->input_value(0), replacement);
    EXPECT_EQ(replacement.get_node()->input_value(0), input->output(0));
    EXPECT_EQ(replacement.get_node()->input_value(1).get_shape(), ov::Shape{4});
    expect_identical(evaluate(model, values), expected);
}

TEST_P(GGUFHadamardSignFolding, SkipsDifferentSignsUnrotatedAndUnmarkedConsumers) {
    const auto type = GetParam();
    for (int consumer = 0; consumer < 3; ++consumer) {
        SCOPED_TRACE(consumer);
        const auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2, 4});
        const auto scaled =
            std::make_shared<Multiply>(input, Constant::create(type, {4}, {0.75f, -0.375f, 2.0f, 3.0f}));
        const auto first = marked_signs(scaled, Constant::create(type, {4}, {-1, 1, -1, 1}));
        ov::Output<ov::Node> other = scaled;
        if (consumer == 0)
            other = marked_signs(scaled, Constant::create(type, {4}, {-1, 1, 1, 1}));
        if (consumer == 2)
            other = std::make_shared<Multiply>(scaled, Constant::create(type, {4}, {-1, 1, -1, 1}));
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{first, other}, ov::ParameterVector{input});
        const auto values = input_tensor(type);
        const auto expected = evaluate(model, values);

        fold_hadamard_input_signs(model);

        EXPECT_EQ(model->get_results()[0]->input_value(0), first->output(0));
        EXPECT_EQ(first->input_value(0), scaled->output(0));
        EXPECT_EQ(model->get_results()[1]->input_value(0), other);
        expect_identical(evaluate(model, values), expected);
    }
}

TEST_P(GGUFHadamardSignFolding, SkipsNonUnitSigns) {
    const auto type = GetParam();
    for (const float invalid : {0.0f, 0.5f, 2.0f, std::numeric_limits<float>::quiet_NaN()}) {
        SCOPED_TRACE(invalid);
        const auto input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2, 4});
        const auto scaled = std::make_shared<Multiply>(input, Constant::create(type, {}, {0.375f}));
        const auto rotation = marked_signs(scaled, Constant::create(type, {4}, {-1.0f, 1.0f, invalid, 1.0f}));
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{rotation}, ov::ParameterVector{input});

        fold_hadamard_input_signs(model);

        EXPECT_EQ(model->get_results()[0]->input_value(0), rotation->output(0));
        EXPECT_EQ(rotation->input_value(0), scaled->output(0));
    }
}

TEST(GGUFHadamardSignFoldingConstants, ExactConvertAndConvertLikeKeepEffectivePrecision) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 4});
    input->output(0).set_names({"x"});
    const auto scale_source = Constant::create(ov::element::f16, {4}, {0.75f, -0.375f, 2.0f, 3.0f});
    const auto scale = std::make_shared<ov::op::v0::Convert>(scale_source, ov::element::f32);
    const auto scaled = std::make_shared<Multiply>(input, scale);
    const auto sign_source = Constant::create(ov::element::i8, {4}, {-1, 1, -1, 1});
    const auto signs = std::make_shared<ov::op::v1::ConvertLike>(sign_source, input);
    const auto rotation = marked_signs(scaled, signs);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{rotation}, ov::ParameterVector{input});
    const auto values = input_tensor(ov::element::f32);
    const auto expected = ov_gguf_test::run_on_cpu(model, {{"x", values}});

    fold_hadamard_input_signs(model);

    const auto replacement = ov::as_type_ptr<Multiply>(model->get_results()[0]->input_value(0).get_node_shared_ptr());
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(replacement->input_value(0), input->output(0));
    const auto folded = ov::as_type_ptr<Constant>(replacement->input_value(1).get_node_shared_ptr());
    ASSERT_NE(folded, nullptr);
    EXPECT_EQ(folded->get_element_type(), ov::element::f32);
    EXPECT_EQ(folded->cast_vector<float>(), (std::vector<float>{-0.75f, -0.375f, -2, 3}));
    expect_identical(ov_gguf_test::run_on_cpu(model, {{"x", values}}), expected);
}

TEST(GGUFHadamardSignFoldingConstants, SkipsLossyConvert) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{2, 4});
    const auto source = Constant::create(ov::element::f32, {4}, {0.1f, 0.2f, 0.3f, 0.4f});
    const auto scale = std::make_shared<ov::op::v0::Convert>(source, ov::element::f16);
    const auto scaled = std::make_shared<Multiply>(input, scale);
    const auto rotation = marked_signs(scaled, Constant::create(ov::element::f16, {4}, {-1, 1, -1, 1}));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{rotation}, ov::ParameterVector{input});

    fold_hadamard_input_signs(model);

    EXPECT_EQ(model->get_results()[0]->input_value(0), rotation->output(0));
    EXPECT_EQ(rotation->input_value(0), scaled->output(0));
}

TEST(GGUFHadamardSignFoldingShapes, PreservesDynamicBatchWithoutExpandingShapes) {
    for (const bool expansion : {false, true}) {
        SCOPED_TRACE(expansion);
        const auto input = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            expansion ? ov::PartialShape{ov::Dimension::dynamic(), 1} : ov::PartialShape{ov::Dimension::dynamic(), 4});
        const auto scaled = std::make_shared<Multiply>(input, Constant::create(ov::element::f32, {}, {0.375f}));
        const auto rotation = marked_signs(scaled, Constant::create(ov::element::f32, {4}, {-1, 1, -1, 1}));
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{rotation}, ov::ParameterVector{input});

        fold_hadamard_input_signs(model);

        if (expansion) {
            EXPECT_EQ(model->get_results()[0]->input_value(0), rotation->output(0));
        } else {
            const auto replacement = model->get_results()[0]->input_value(0);
            EXPECT_EQ(replacement.get_node()->input_value(0), input->output(0));
            EXPECT_EQ(replacement.get_partial_shape(), input->get_partial_shape());
            const auto values = input_tensor(ov::element::f32);
            const auto expected =
                marked_signs(std::make_shared<Multiply>(input, Constant::create(ov::element::f32, {}, {0.375f})),
                             Constant::create(ov::element::f32, {4}, {-1, 1, -1, 1}));
            const auto reference = std::make_shared<ov::Model>(ov::OutputVector{expected}, ov::ParameterVector{input});
            ov::TensorVector before{ov::Tensor(ov::element::f32, {2, 4})};
            ov::TensorVector after{ov::Tensor(ov::element::f32, {2, 4})};
            ASSERT_TRUE(reference->evaluate(before, {values}));
            ASSERT_TRUE(model->evaluate(after, {values}));
            expect_identical(after, before);
        }
    }
}

TEST(GGUFHadamardSignFoldingShapes, SingletonSqueezeKeepsDynamicTokensNumericsAndSharedSigns) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{ov::Dimension::dynamic(), 1, ov::Dimension::dynamic(), 4});
    input->output(0).set_names({"x"});
    const auto scaled =
        std::make_shared<Multiply>(input, Constant::create(ov::element::f32, {4}, {-0.75f, 1.375f, 0.25f, -2.0f}));
    const auto squeezed = std::make_shared<ov::op::v0::Squeeze>(scaled, Constant::create(ov::element::i64, {1}, {1}));
    const auto first = marked_signs(squeezed, Constant::create(ov::element::f32, {4}, {-1, 1, -1, 1}));
    const auto second = marked_signs(squeezed, Constant::create(ov::element::f32, {1, 1, 4}, {-1, 1, -1, 1}));
    const auto direct = marked_signs(scaled, Constant::create(ov::element::f32, {1, 1, 1, 4}, {-1, 1, -1, 1}));
    const auto source_shape = std::make_shared<ov::op::v3::ShapeOf>(scaled);
    const auto view_shape = std::make_shared<ov::op::v3::ShapeOf>(squeezed);
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{first, second, direct, source_shape, view_shape},
                                                   ov::ParameterVector{input});
    std::vector<float> data(24);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = (static_cast<float>(i) - 11.0f) * 0.375f;
    const auto values = ov_gguf_test::make_f32_tensor({2, 1, 3, 4}, data);
    const auto outputs = [] {
        return ov::TensorVector{ov::Tensor(ov::element::f32, {2, 3, 4}),
                                ov::Tensor(ov::element::f32, {2, 3, 4}),
                                ov::Tensor(ov::element::f32, {2, 1, 3, 4}),
                                ov::Tensor(ov::element::i64, {4}),
                                ov::Tensor(ov::element::i64, {3})};
    };
    auto expected = outputs();
    ASSERT_TRUE(model->evaluate(expected, {values}));
    const auto cpu_expected = ov_gguf_test::run_on_cpu(model, {{"x", values}});

    fold_hadamard_input_signs(model);

    const auto replacement = model->get_results()[0]->input_value(0);
    ASSERT_TRUE(ov::is_type<ov::op::v0::Squeeze>(replacement.get_node()));
    EXPECT_EQ(model->get_results()[1]->input_value(0), replacement);
    const auto new_scaled = replacement.get_node()->input_value(0);
    ASSERT_TRUE(ov::is_type<Multiply>(new_scaled.get_node()));
    EXPECT_EQ(new_scaled.get_node()->input_value(0), input->output(0));
    EXPECT_EQ(model->get_results()[2]->input_value(0), new_scaled);
    EXPECT_EQ(source_shape->input_value(0), new_scaled);
    EXPECT_EQ(view_shape->input_value(0), replacement);
    EXPECT_EQ(replacement.get_partial_shape(), squeezed->get_output_partial_shape(0));
    auto actual = outputs();
    ASSERT_TRUE(model->evaluate(actual, {values}));
    expect_identical(actual, expected);
    expect_identical(ov_gguf_test::run_on_cpu(model, {{"x", values}}), cpu_expected);
}

TEST(GGUFHadamardSignFoldingShapes, DifferentSignsOrUnrotatedSqueezeViewSkipWholeGroup) {
    for (int consumer = 0; consumer < 3; ++consumer) {
        SCOPED_TRACE(consumer);
        const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 4});
        const auto scaled = std::make_shared<Multiply>(input, Constant::create(ov::element::f32, {4}, {0.375f}));
        const auto axes = Constant::create(ov::element::i64, {1}, {1});
        const auto squeezed = std::make_shared<ov::op::v0::Squeeze>(scaled, axes);
        const auto first = marked_signs(squeezed, Constant::create(ov::element::f32, {4}, {-1, 1, -1, 1}));
        ov::Output<ov::Node> other = squeezed;
        if (consumer == 0)
            other = marked_signs(squeezed, Constant::create(ov::element::f32, {4}, {-1, 1, 1, 1}));
        if (consumer == 1) {
            const auto view = std::make_shared<ov::op::v0::Squeeze>(scaled, axes);
            other = marked_signs(view, Constant::create(ov::element::f32, {4}, {-1, 1, 1, 1}));
        }
        const auto model = std::make_shared<ov::Model>(ov::OutputVector{first, other}, ov::ParameterVector{input});
        const auto values = ov_gguf_test::make_f32_tensor({2, 1, 4}, {-1, 2, -3, 4, 5, -6, 7, -8});
        const auto expected = evaluate(model, values);

        fold_hadamard_input_signs(model);

        EXPECT_EQ(model->get_results()[0]->input_value(0), first->output(0));
        EXPECT_EQ(first->input_value(0), squeezed->output(0));
        EXPECT_EQ(squeezed->input_value(0), scaled->output(0));
        EXPECT_EQ(model->get_results()[1]->input_value(0), other);
        expect_identical(evaluate(model, values), expected);
    }
}

INSTANTIATE_TEST_SUITE_P(GGUF,
                         GGUFHadamardSignFolding,
                         ::testing::Values(ov::element::f16, ov::element::f32),
                         [](const ::testing::TestParamInfo<ov::element::Type>& value) {
                             return value.param.get_type_name();
                         });
}  // namespace
