// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mlp_seq.hpp"

#include "snippets/op/subgraph.hpp"

#include "openvino/opsets/opset15.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> MLPSeqFunction::initOriginal() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    std::shared_ptr<Node> current = A_param;
    float const_value = 0.1122f;

    auto mlp_layer = [&](size_t m, size_t n) {
        auto b_shape = ov::Shape{m, n};
        auto b_row = ov::Shape{m};
        auto B = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_shape, const_value);
        current = std::make_shared<ov::op::v0::MatMul>(current, B, false, true);
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, const_value);
        current = std::make_shared<ov::op::v1::Add>(current, constant);
        current = std::make_shared<ov::op::v0::Relu>(current);

        const_value += 0.1122f;
    };

    mlp_layer(hidden_matmul_size, static_cast<unsigned long>(input_shapes[0][1].get_length()));
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        mlp_layer(hidden_matmul_size, hidden_matmul_size);
    }
    mlp_layer(static_cast<unsigned long>(input_shapes[0][1].get_length()), hidden_matmul_size);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A_param});
}

std::shared_ptr<ov::Model> MLPSeqQuantizedFunction::initOriginal() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    std::shared_ptr<Node> current = A_param;

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::f32};
    float const_value = 0.1122f;

    auto mlp_layer = [&](size_t m, size_t n) {
        auto b_shape = ov::Shape{m, n};
        auto b_row = ov::Shape{m};
        std::shared_ptr<Node> B = std::make_shared<ov::op::v0::Constant>(ov::element::i8, b_shape, const_value);
        auto fp_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_shape, const_value);
        B = std::make_shared<ov::op::v0::Convert>(B, ov::element::f32);
        B = std::make_shared<ov::op::v1::Multiply>(B, fp_const);
        current = ov::builder::subgraph::makeFakeQuantize(current, ov::element::f32, onData);
        current = std::make_shared<ov::op::v0::MatMul>(current, B, false, true);
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, const_value);
        current = std::make_shared<ov::op::v1::Add>(current, constant);
        current = std::make_shared<ov::op::v0::Relu>(current);

        const_value += 0.1122f;
    };

    mlp_layer(hidden_matmul_size, static_cast<unsigned long>(input_shapes[0][1].get_length()));
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        mlp_layer(hidden_matmul_size, hidden_matmul_size);
    }
    mlp_layer(static_cast<unsigned long>(input_shapes[0][1].get_length()), hidden_matmul_size);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);

    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A_param});
}

std::shared_ptr<ov::Model> MLPSeqQuantizedTypeRelaxedFunction::initOriginal() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    std::shared_ptr<Node> A = A_param;
    if (precisions[0] != ov::element::u8) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::u8);
    }

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8};
    std::shared_ptr<Node> current = A;
    float const_value = 0.1122f;

    auto mlp_layer = [&](size_t m, size_t n) {
        auto b_shape = ov::Shape{m, n};
        auto b_row = ov::Shape{m};
        current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, ov::element::f32, onData);
        auto B = std::make_shared<ov::op::v0::Constant>(ov::element::i8, b_shape, const_value);

        current = std::make_shared<op::TypeRelaxed<ov::op::v0::MatMul>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(B, element::f32).get(),
            false,
            true);

        auto dq_scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, const_value);

        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(dq_scale, element::f32).get());
        auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, const_value);
        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(constant, element::f32).get());
        current = std::make_shared<ov::op::v0::Relu>(current);

        const_value += 0.1122f;
    };

    mlp_layer(hidden_matmul_size, static_cast<unsigned long>(input_shapes[0][1].get_length()));
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        mlp_layer(hidden_matmul_size, hidden_matmul_size);
    }
    mlp_layer(static_cast<unsigned long>(input_shapes[0][1].get_length()), hidden_matmul_size);

    ov::builder::subgraph::FakeQuantizeOnData onData2 =
        {256, {1, 1}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::f32};
    current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, ov::element::f32, onData2);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);
    ov::builder::subgraph::FakeQuantizeOnData onData1 =
        {256, {1, 1}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::f32};
    auto dq_A = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(softmax, ov::element::f32, onData1);

    auto result = std::make_shared<ov::op::v0::Result>(dq_A);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A_param});
}

std::shared_ptr<ov::Model> MLPSeqQuantizedTypeRelaxedFunction::initReference() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    std::shared_ptr<Node> A = A_param;
    if (precisions[0] != ov::element::u8) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::u8);
    }

    // Create subgraph parameters (must be preserved even if similar constants exist).
    auto sub_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    ov::ParameterVector subgraph_params = {sub_A};

    ov::NodeVector subgraph_nodes = {A};

    ov::builder::subgraph::FakeQuantizeOnData onData = {
        256, {1, 1}, {0.0f}, {2.55f}, {0.f}, {255.f}, ov::element::u8
    };

    auto decomposed_fq = [](const ov::Output<ov::Node>& input,
                            const ov::element::Type& out_precision,
                            float il, float ih, float scale) -> std::shared_ptr<ov::Node> {
        auto input_low = ov::op::v0::Constant::create(input.get_element_type(), {1, 1}, {il});
        auto input_high = ov::op::v0::Constant::create(input.get_element_type(), {1, 1}, {ih});
        auto output_scale = ov::op::v0::Constant::create(input.get_element_type(), {1, 1}, {scale});
        auto max_node = std::make_shared<ov::op::v1::Maximum>(input, input_low);
        auto min_node = std::make_shared<ov::op::v1::Minimum>(max_node, input_high);
        return std::make_shared<ov::op::v1::Multiply>(min_node, output_scale);
    };

    std::shared_ptr<ov::Node> current = sub_A;
    current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::f32);

    auto mlp_layer = [&](size_t m, size_t n, bool trans) {
        auto b_shape = ov::Shape{m, n};
        auto b_row = ov::Shape{m};
        auto b_shape_trans = ov::Shape{n, m};

        auto dq_shifts = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, b_row);
        auto dq_shifts_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, std::vector<float>{0.1122f});
        auto dq_scales = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, b_row);
        auto dq_scales_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, b_row, std::vector<float>{0.1122f});

        auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, b_shape_trans);
        auto B_const = std::make_shared<ov::op::v0::Constant>(ov::element::i8, b_shape, std::vector<float>{0.1122f});
        auto B_const_trans =
            std::make_shared<ov::op::v1::Transpose>(B_const,
                                                    ov::op::v0::Constant::create(ov::element::i32, {2}, {1, 0}));
        subgraph_params.push_back(B);
        subgraph_nodes.push_back(B_const_trans);

        current = decomposed_fq(current,
                                ov::element::u8,
                                onData.inputLowValues[0],
                                onData.inputHighValues[0],
                                0.00346764503f);
        current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::u8);

        current = std::make_shared<op::TypeRelaxed<ov::op::v0::MatMul>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(B, ov::element::f32).get(),
            false,
            trans);

        subgraph_params.push_back(dq_scales);
        subgraph_nodes.push_back(dq_scales_const);
        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(dq_scales, ov::element::f32).get());

        subgraph_params.push_back(dq_shifts);
        subgraph_nodes.push_back(dq_shifts_const);
        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(dq_shifts, ov::element::f32).get());
        current = std::make_shared<ov::op::v0::Relu>(current);
    };

    mlp_layer(hidden_matmul_size, static_cast<unsigned long>(input_shapes[0][1].get_length()), false);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        mlp_layer(hidden_matmul_size, hidden_matmul_size, true);
    }
    mlp_layer(static_cast<unsigned long>(input_shapes[0][1].get_length()), hidden_matmul_size, false);

    current = decomposed_fq(current, ov::element::f32, onData.inputLowValues[0], onData.inputHighValues[0], 0.00346764503f);
    current = std::make_shared<ov::op::v1::Subtract>(current, ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {0}));
    current = std::make_shared<ov::op::v5::Round>(current, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    current = std::make_shared<ov::op::v8::Softmax>(current, 1);
    current = decomposed_fq(current, ov::element::f32, onData.inputLowValues[0], onData.inputHighValues[0], 0.00346764503f);
    current = std::make_shared<ov::op::v1::Subtract>(current, ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {0}));
    current = std::make_shared<ov::op::v5::Round>(current, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto result_subgraph = std::make_shared<ov::op::v0::Result>(current);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        subgraph_nodes,
        std::make_shared<ov::Model>(ov::ResultVector{result_subgraph}, subgraph_params));
    auto result = std::make_shared<ov::op::v0::Result>(subgraph);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{A_param});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
