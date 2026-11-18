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
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/rt_info/decompression.hpp"

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

struct UpSignGraph {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<Constant> codes;
    std::shared_ptr<Constant> scale;
    std::shared_ptr<Constant> zero_point;
    std::shared_ptr<ov::op::v0::Convert> code_convert;
    std::shared_ptr<ov::op::v1::Subtract> subtract;
    std::shared_ptr<Multiply> scaled;
    std::shared_ptr<ov::op::v1::Reshape> weight_view;
    std::shared_ptr<ov::Node> weights;
    std::shared_ptr<ov::op::v0::MatMul> up;
    std::shared_ptr<Multiply> swiglu;
    std::shared_ptr<ov::op::v1::Reshape> view;
    std::shared_ptr<Multiply> rotation;
    std::shared_ptr<ov::op::v0::MatMul> hadamard;
    std::shared_ptr<ov::Model> model;
};

UpSignGraph up_sign_graph(const ov::element::Type& type, size_t columns = 4, bool dynamic_tokens = false) {
    UpSignGraph graph;
    graph.input = std::make_shared<ov::op::v0::Parameter>(
        type,
        ov::PartialShape{dynamic_tokens ? ov::Dimension::dynamic() : ov::Dimension(2), ov::Dimension(columns)});
    graph.input->output(0).set_names({"x"});
    std::vector<uint8_t> codes(4 * columns);
    for (size_t i = 0; i < codes.size(); ++i)
        codes[i] = static_cast<uint8_t>((i + i / columns) % 3);
    graph.codes = Constant::create(ov::element::u2, {4, 1, columns}, codes);
    graph.scale = Constant::create(ov::element::f16, {4, 1, 1}, {0.25f, -0.5f, 0.75f, 1.25f});
    graph.zero_point = Constant::create(ov::element::f16, {}, {1});
    graph.code_convert = std::make_shared<ov::op::v0::Convert>(graph.codes, ov::element::f16);
    ov::mark_as_decompression(graph.code_convert);
    graph.subtract = std::make_shared<ov::op::v1::Subtract>(graph.code_convert, graph.zero_point);
    graph.scaled = std::make_shared<Multiply>(graph.subtract, graph.scale);
    graph.weight_view = std::make_shared<ov::op::v1::Reshape>(
        graph.scaled,
        Constant::create(ov::element::i64, {2}, std::vector<int64_t>{4, static_cast<int64_t>(columns)}),
        false);
    graph.weights = std::make_shared<ov::op::v0::Convert>(graph.weight_view, type);
    graph.up = std::make_shared<ov::op::v0::MatMul>(graph.input, graph.weights, false, true);
    const auto gate_weights = Constant::create(type, {4, columns}, {0.125f});
    const auto gate = std::make_shared<ov::op::v0::MatMul>(graph.input, gate_weights, false, true);
    const auto swish = std::make_shared<ov::op::v4::Swish>(gate);
    graph.swiglu = std::make_shared<Multiply>(swish, graph.up);
    const auto expanded =
        std::make_shared<ov::op::v1::Reshape>(graph.swiglu, Constant::create(ov::element::i64, {3}, {-1, 1, 4}), false);
    graph.view =
        std::make_shared<ov::op::v1::Reshape>(expanded, Constant::create(ov::element::i64, {2}, {-1, 4}), false);
    graph.rotation = marked_signs(graph.view, Constant::create(type, {1, 4}, {-1, 1, -1, 1}), true);
    graph.hadamard = std::make_shared<ov::op::v0::MatMul>(
        graph.rotation,
        Constant::create(type, {4, 4}, {1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1}));
    graph.model = std::make_shared<ov::Model>(ov::OutputVector{graph.hadamard}, ov::ParameterVector{graph.input});
    return graph;
}

std::shared_ptr<ov::op::v0::MatMul> folded_up(const UpSignGraph& graph) {
    auto data = graph.hadamard->input_value(0);
    while (ov::is_type<ov::op::v1::Reshape>(data.get_node()) || ov::is_type<ov::op::v0::Squeeze>(data.get_node()) ||
           ov::is_type<ov::op::v0::Unsqueeze>(data.get_node()))
        data = data.get_node()->input_value(0);
    const auto swiglu = ov::as_type_ptr<Multiply>(data.get_node_shared_ptr());
    return swiglu ? ov::as_type_ptr<ov::op::v0::MatMul>(swiglu->get_input_node_shared_ptr(1)) : nullptr;
}

void expect_same_floats(const ov::Tensor& actual, const ov::Tensor& expected, float tolerance = 0) {
    ASSERT_EQ(actual.get_element_type(), expected.get_element_type());
    ASSERT_EQ(actual.get_shape(), expected.get_shape());
    const auto value = [](const ov::Tensor& tensor, size_t i) {
        return tensor.get_element_type() == ov::element::f16 ? static_cast<float>(tensor.data<const ov::float16>()[i])
                                                             : tensor.data<const float>()[i];
    };
    for (size_t i = 0; i < actual.get_size(); ++i)
        EXPECT_NEAR(value(actual, i), value(expected, i), tolerance) << i;
}

TEST_P(GGUFHadamardSignFolding, SwiGLURowSignsPreservePackedWeightsNumericsAndMetadata) {
    auto graph = up_sign_graph(GetParam());
    graph.code_convert->get_rt_info()["code_metadata"] = 17;
    graph.code_convert->output(0).get_rt_info()["code_output_metadata"] = 18;
    graph.rotation->get_rt_info()["rotation_metadata"] = 19;
    graph.rotation->output(0).get_rt_info()["rotation_output_metadata"] = 20;
    graph.rotation->output(0).set_names({"rotated"});
    const auto values = input_tensor(GetParam());
    const auto expected = evaluate(graph.model, values);
    const auto cpu_expected = ov_gguf_test::run_on_cpu(graph.model, {{"x", values}});
    const auto original_codes = graph.codes->cast_vector<uint8_t>();

    fold_hadamard_input_signs(graph.model);

    const auto up = folded_up(graph);
    ASSERT_NE(up, nullptr);
    EXPECT_NE(up, graph.up);
    const auto weight_convert = ov::as_type_ptr<ov::op::v0::Convert>(up->get_input_node_shared_ptr(1));
    ASSERT_NE(weight_convert, nullptr);
    const auto weight_view = ov::as_type_ptr<ov::op::v1::Reshape>(weight_convert->get_input_node_shared_ptr(0));
    ASSERT_NE(weight_view, nullptr);
    const auto scaled = ov::as_type_ptr<Multiply>(weight_view->get_input_node_shared_ptr(0));
    ASSERT_NE(scaled, nullptr);
    const auto scale = ov::as_type_ptr<Constant>(scaled->get_input_node_shared_ptr(1));
    ASSERT_NE(scale, nullptr);
    EXPECT_EQ(scale->get_element_type(), ov::element::f16);
    EXPECT_EQ(scale->cast_vector<float>(), (std::vector<float>{-0.25f, -0.5f, -0.75f, 1.25f}));
    const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(scaled->get_input_node_shared_ptr(0));
    ASSERT_NE(subtract, nullptr);
    EXPECT_EQ(subtract->input_value(1), graph.zero_point->output(0));
    const auto code_convert = ov::as_type_ptr<ov::op::v0::Convert>(subtract->get_input_node_shared_ptr(0));
    ASSERT_NE(code_convert, nullptr);
    EXPECT_TRUE(ov::is_decompression(code_convert));
    EXPECT_EQ(code_convert->get_rt_info().at("code_metadata").as<int>(), 17);
    EXPECT_EQ(code_convert->output(0).get_rt_info().at("code_output_metadata").as<int>(), 18);
    const auto codes = ov::as_type_ptr<Constant>(code_convert->get_input_node_shared_ptr(0));
    ASSERT_NE(codes, nullptr);
    EXPECT_EQ(codes, graph.codes);
    EXPECT_EQ(codes->get_element_type(), ov::element::u2);
    EXPECT_EQ(codes->get_shape(), graph.codes->get_shape());
    EXPECT_EQ(codes->cast_vector<uint8_t>(), original_codes);
    EXPECT_EQ(graph.codes->cast_vector<uint8_t>(), original_codes);
    const auto replacement = graph.hadamard->input_value(0);
    EXPECT_EQ(replacement.get_node()->get_rt_info().at("rotation_metadata").as<int>(), 19);
    EXPECT_EQ(replacement.get_rt_info().at("rotation_output_metadata").as<int>(), 20);
    EXPECT_EQ(replacement.get_names().count("rotated"), 1u);
    EXPECT_EQ(replacement.get_node()->get_rt_info().count("gguf_hadamard_input_sign"), 0u);
    expect_same_floats(evaluate(graph.model, values)[0], expected[0]);
    expect_same_floats(ov_gguf_test::run_on_cpu(graph.model, {{"x", values}}), cpu_expected, 0.002f);
}

TEST(GGUFHadamardSignFoldingUp, SharedConsumersRemainUnrotatedAndNonByteAlignedRowsFold) {
    auto graph = up_sign_graph(ov::element::f32, 3);
    const auto other_signs = marked_signs(graph.view, Constant::create(ov::element::f32, {4}, {1, -1, 1, -1}));
    graph.model = std::make_shared<ov::Model>(
        ov::OutputVector{graph.hadamard, graph.up, graph.swiglu, graph.weights, other_signs},
        ov::ParameterVector{graph.input});
    const auto values = ov_gguf_test::make_f32_tensor({2, 3}, {1.125f, -2.5f, 3.75f, -4.25f, -5.5f, 6.125f});
    const auto expected = evaluate(graph.model, values);

    fold_hadamard_input_signs(graph.model);

    EXPECT_NE(graph.hadamard->input_value(0), graph.rotation->output(0));
    EXPECT_EQ(graph.model->get_results()[1]->input_value(0), graph.up->output(0));
    EXPECT_EQ(graph.model->get_results()[2]->input_value(0), graph.swiglu->output(0));
    EXPECT_EQ(graph.model->get_results()[3]->input_value(0), graph.weights->output(0));
    EXPECT_NE(graph.model->get_results()[4]->input_value(0), other_signs->output(0));
    const auto actual = evaluate(graph.model, values);
    for (size_t i = 0; i < actual.size(); ++i)
        expect_same_floats(actual[i], expected[i]);
}

TEST(GGUFHadamardSignFoldingUp, SingletonUnsqueezeAndSqueezePreserveNumerics) {
    auto graph = up_sign_graph(ov::element::f32);
    const auto axes = Constant::create(ov::element::i64, {1}, {1});
    const auto expanded = std::make_shared<ov::op::v0::Unsqueeze>(graph.swiglu, axes);
    const auto squeezed = std::make_shared<ov::op::v0::Squeeze>(expanded, axes);
    graph.view->set_argument(0, squeezed);
    graph.model->validate_nodes_and_infer_types();
    const auto values = input_tensor(ov::element::f32);
    const auto expected = evaluate(graph.model, values);

    fold_hadamard_input_signs(graph.model);

    ASSERT_NE(folded_up(graph), nullptr);
    EXPECT_NE(graph.hadamard->input_value(0), graph.rotation->output(0));
    expect_same_floats(evaluate(graph.model, values)[0], expected[0]);
}

TEST(GGUFHadamardSignFoldingUp, DynamicShapeQueriesDoNotKeepUnrotatedProjectionsAlive) {
    auto graph = up_sign_graph(ov::element::f32, 4, true);
    const auto up_shape = std::make_shared<ov::op::v0::ShapeOf>(graph.up);
    const auto glu_shape = std::make_shared<ov::op::v3::ShapeOf>(graph.swiglu, ov::element::i64);
    const auto view_shape = std::make_shared<ov::op::v3::ShapeOf>(graph.view, ov::element::i64);
    const auto restored = std::make_shared<ov::op::v1::Reshape>(graph.hadamard, view_shape, false);
    graph.model = std::make_shared<ov::Model>(ov::OutputVector{restored, up_shape, glu_shape, view_shape},
                                              ov::ParameterVector{graph.input});
    const auto values = input_tensor(ov::element::f32);
    ov::TensorVector expected{ov::Tensor(ov::element::f32, {2, 4}),
                              ov::Tensor(ov::element::i64, {2}),
                              ov::Tensor(ov::element::i64, {2}),
                              ov::Tensor(ov::element::i64, {2})};
    ASSERT_TRUE(graph.model->evaluate(expected, {values}));

    fold_hadamard_input_signs(graph.model);

    ASSERT_NE(folded_up(graph), nullptr);
    EXPECT_NE(graph.hadamard->input_value(0), graph.rotation->output(0));
    EXPECT_EQ(graph.hadamard->get_output_partial_shape(0), (ov::PartialShape{-1, 4}));
    EXPECT_NE(up_shape->get_input_node_shared_ptr(0), graph.up);
    EXPECT_NE(glu_shape->get_input_node_shared_ptr(0), graph.swiglu);
    EXPECT_NE(view_shape->get_input_node_shared_ptr(0), graph.view);
    const auto operations = graph.model->get_ordered_ops();
    EXPECT_EQ(std::count_if(operations.begin(),
                            operations.end(),
                            [](const std::shared_ptr<ov::Node>& node) {
                                return ov::is_type<ov::op::v0::MatMul>(node);
                            }),
              3);
    ov::TensorVector actual{ov::Tensor(ov::element::f32, {2, 4}),
                            ov::Tensor(ov::element::i64, {2}),
                            ov::Tensor(ov::element::i64, {2}),
                            ov::Tensor(ov::element::i64, {2})};
    ASSERT_TRUE(graph.model->evaluate(actual, {values}));
    expect_same_floats(actual[0], expected[0]);
    for (size_t i = 1; i < actual.size(); ++i)
        expect_identical(actual[i], expected[i]);
}

TEST(GGUFHadamardSignFoldingUp, InvalidCodesZeroPointsScalesSignsViewsConversionsAndMetadataSkip) {
    for (int invalid = 0; invalid < 12; ++invalid) {
        SCOPED_TRACE(invalid);
        auto graph = up_sign_graph(ov::element::f32);
        if (invalid == 0)
            graph.code_convert->set_argument(0, Constant::create(ov::element::u2, {4, 1, 4}, {3}));
        if (invalid == 1)
            graph.subtract->set_argument(1, Constant::create(ov::element::f16, {}, {0}));
        if (invalid == 2) {
            ov::Tensor scale(ov::element::f16, {4, 1, 1});
            std::fill_n(scale.data<ov::float16>(),
                        scale.get_size(),
                        ov::float16(std::numeric_limits<float>::infinity()));
            graph.scaled->set_argument(1, std::make_shared<Constant>(scale));
        }
        if (invalid == 3)
            graph.rotation->set_argument(0, Constant::create(ov::element::f32, {1, 4}, {-1.0f, 1.0f, 0.5f, 1.0f}));
        if (invalid == 4) {
            const auto mixed = std::make_shared<ov::op::v1::Reshape>(graph.swiglu,
                                                                     Constant::create(ov::element::i64, {2}, {-1, 2}),
                                                                     false);
            graph.view->set_argument(0, mixed);
        }
        if (invalid == 5)
            graph.rotation->set_argument(0, Constant::create(ov::element::f32, {2, 4}, {-1, 1, -1, 1, 1, -1, 1, -1}));
        if (invalid == 6) {
            const auto narrowed = std::make_shared<ov::op::v0::Convert>(graph.weight_view, ov::element::bf16);
            graph.up->set_argument(1, std::make_shared<ov::op::v0::Convert>(narrowed, ov::element::f32));
        }
        if (invalid == 7)
            graph.rotation->get_rt_info().erase("gguf_hadamard_input_sign");
        if (invalid == 8) {
            graph.view->get_rt_info()["conflict"] = 1;
            graph.rotation->get_rt_info()["conflict"] = 2;
        }
        if (invalid == 9) {
            graph.zero_point = Constant::create(ov::element::u8, {}, {1});
            graph.subtract->set_argument(1, std::make_shared<ov::op::v0::Convert>(graph.zero_point, ov::element::f16));
            graph.code_convert->set_argument(0, Constant::create(ov::element::u4, {4, 1, 4}, {1}));
        }
        if (invalid == 10)
            graph.up->set_transpose_b(false);
        if (invalid == 11)
            graph.swiglu->set_argument(0, graph.up);
        graph.model->validate_nodes_and_infer_types();
        const auto values = input_tensor(ov::element::f32);
        ov::TensorVector expected;
        if (invalid != 2)
            expected = evaluate(graph.model, values);

        fold_hadamard_input_signs(graph.model);

        EXPECT_EQ(graph.hadamard->input_value(0), graph.rotation->output(0));
        EXPECT_EQ(graph.up->input_value(0), graph.input->output(0));
        if (invalid != 2)
            expect_same_floats(evaluate(graph.model, values)[0], expected[0]);
    }
}

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
