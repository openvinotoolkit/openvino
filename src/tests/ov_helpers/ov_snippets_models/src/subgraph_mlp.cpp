// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mlp.hpp"

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
    std::shared_ptr<Node> A = A_param;
    if (precisions[0] != ov::element::f32) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::f32);
    }
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      input_shapes[0].to_shape(),
                                                      std::vector<float>{0.1122});
    std::shared_ptr<Node> current = A;

    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        auto B = std::make_shared<ov::op::v0::Constant>(ov::element::f32, input_shapes[0].to_shape(), std::vector<float>{0.1122f + mm_count});
        current = std::make_shared<ov::op::v0::MatMul>(current, B, false, true);
        current = std::make_shared<ov::op::v1::Multiply>(current, add);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                   ov::Shape{input_shapes[0].to_shape()[0]},
                                                                   std::vector<float>{0.1122f + i});
            current = std::make_shared<ov::op::v1::Add>(current, constant);
        }
    }
    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);
    auto result = std::make_shared<ov::op::v0::Result>(softmax);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A_param});
}

std::shared_ptr<ov::Model> MLPSeqFunction::initReference() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    std::shared_ptr<Node> A = A_param;
    if (precisions[0] != ov::element::f32) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::f32);
    }

    std::vector<std::shared_ptr<ov::Node>> constants;
    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        constants.push_back(std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, input_shapes[0].to_shape(), std::vector<float>{0.1122f + mm_count}));
    }

    std::vector<std::shared_ptr<ov::Node>> transposes;
    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        transposes.push_back(
            std::make_shared<ov::op::v1::Transpose>(
                constants[mm_count],
                ov::op::v0::Constant::create(ov::element::i32, {input_shapes[0].get_shape().size()}, std::vector<int64_t>{1, 0})));
    }

    auto zeros_matrix = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32,
        input_shapes[0].to_shape(),
        std::vector<float>{0.1122});

    std::vector<std::shared_ptr<ov::Node>> zero_vectors;
    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        zero_vectors.push_back(std::make_shared<ov::op::v0::Constant>(
            ov::element::f32,
            ov::Shape{input_shapes[0].to_shape()[0]},
            std::vector<float>{0.1122f + mm_count}));
    }

    // Create subgraph parameters (must be preserved even if similar constants exist).
    auto sub_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    auto sub_trans_zeros0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    auto sub_zeros2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    auto sub_zeros64 = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_trans_zeros1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);

    ov::ParameterVector subgraph_params = {
        sub_A, sub_trans_zeros0, sub_zeros2, sub_zeros64, sub_zeros64_2,
        sub_trans_zeros1, sub_zeros2, sub_zeros64_3, sub_zeros64_4
    };

    ov::NodeVector subgraph_nodes = {
        A, transposes[0], zeros_matrix, zero_vectors[0], zero_vectors[1],
        transposes[1], zeros_matrix, zero_vectors[0], zero_vectors[1],
    };

    std::shared_ptr<ov::Node> current = sub_A;
    current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::f32);
    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::u8);
        current = std::make_shared<ov::op::v0::MatMul>(current, sub_trans_zeros0, false, true);
        current = std::make_shared<ov::op::v1::Multiply>(current, sub_zeros2);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122f + i});
            current = std::make_shared<ov::op::v1::Add>(current, i == 0 ? sub_zeros64 : sub_zeros64_2);
        }
    }

    current = std::make_shared<ov::op::v8::Softmax>(current, 1);
    auto result_subgraph = std::make_shared<ov::op::v0::Result>(current);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        subgraph_nodes,
        std::make_shared<ov::Model>(ov::ResultVector{result_subgraph}, subgraph_params));
    auto result = std::make_shared<ov::op::v0::Result>(subgraph);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{A_param});
}

std::shared_ptr<ov::Model> MLPSeqQuantizedTypeRelaxedFunction::initOriginal() const {
    auto A_param = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    std::shared_ptr<Node> A = A_param;
    if (precisions[0] != ov::element::u8) {
        A = std::make_shared<ov::op::v0::Convert>(A, ov::element::u8);
    }
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      input_shapes[0].to_shape(),
                                                      std::vector<float>{0.1122});

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {0.f}, {2.55f}, {0.f}, {255.f}, ov::element::u8};
    std::shared_ptr<Node> current = A;

    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, ov::element::f32, onData);
        auto B = std::make_shared<ov::op::v0::Constant>(ov::element::i8, input_shapes[0].to_shape(), std::vector<float>{0.1122f + mm_count});

        current = std::make_shared<op::TypeRelaxed<ov::op::v0::MatMul>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(B, element::f32).get(),
            false,
            true);

        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(add, element::f32).get());
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                   ov::Shape{input_shapes[0].to_shape()[0]},
                                                                   std::vector<float>{0.1122f + i});
            current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{element::f32},
                ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(constant, element::f32).get());
        }
    }
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

    std::vector<std::shared_ptr<ov::Node>> constants;
    for (size_t mm_count = 0; mm_count < num_hidden_layers; ++mm_count) {
        constants.push_back(std::make_shared<ov::op::v0::Constant>(
            ov::element::i8, input_shapes[0].to_shape(), std::vector<float>{0.1122f + mm_count}));
    }

    std::vector<std::shared_ptr<ov::Node>> transposes;
    for (size_t mm_count = 0; mm_count < num_hidden_layers; ++mm_count) {
        transposes.push_back(
            std::make_shared<ov::op::v1::Transpose>(
                constants[mm_count],
                ov::op::v0::Constant::create(ov::element::i32, {input_shapes[0].get_shape().size()}, std::vector<int64_t>{1, 0})));
    }

    auto zeros_matrix = std::make_shared<ov::op::v0::Constant>(
        ov::element::f32,
        input_shapes[0].to_shape(),
        std::vector<float>{0.1122});

    std::vector<std::shared_ptr<ov::Node>> zero_vectors;
    for (size_t mm_count = 0; mm_count < num_hidden_layers; ++mm_count) {
        zero_vectors.push_back(std::make_shared<ov::op::v0::Constant>(
            ov::element::f32,
            ov::Shape{input_shapes[0].to_shape()[0]},
            std::vector<float>{0.0f + mm_count}));
    }
    std::vector<std::shared_ptr<ov::Node>> hidden_vectors;
    for (size_t mm_count = 0; mm_count < num_hidden_layers; ++mm_count) {
        hidden_vectors.push_back(std::make_shared<ov::op::v0::Constant>(
            ov::element::f32,
            ov::Shape{input_shapes[0].to_shape()[0]},
            std::vector<float>{0.1122f + mm_count}));
    }

    // Create subgraph parameters (must be preserved even if similar constants exist).
    auto sub_A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    auto sub_trans_zeros0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    auto sub_zeros2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> sub_zeros64_vec;
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        sub_zeros64_vec.push_back(std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            ov::Shape{input_shapes[0].to_shape()[0]}));
    }
    std::vector<std::shared_ptr<ov::op::v0::Parameter>> sub_hidden_vec;
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        sub_hidden_vec.push_back(std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            ov::Shape{input_shapes[0].to_shape()[0]}));
    }
    auto sub_trans_zeros1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shapes[0]);

    ov::ParameterVector subgraph_params = {sub_A, sub_trans_zeros0, sub_zeros2};
    for (const auto& param : sub_zeros64_vec) {
        subgraph_params.push_back(param);
    }
    subgraph_params.push_back(sub_trans_zeros1);
    subgraph_params.push_back(sub_zeros2);
    for (const auto& param : sub_hidden_vec) {
        subgraph_params.push_back(param);
    }

    ov::NodeVector subgraph_nodes = {A, transposes[0], zeros_matrix};
    for (const auto& param : zero_vectors) {
        subgraph_nodes.push_back(param);
    }
    subgraph_nodes.push_back(transposes[1]);
    subgraph_nodes.push_back(zeros_matrix);
    for (const auto& param : hidden_vectors) {
        subgraph_nodes.push_back(param);
    }

    ov::builder::subgraph::FakeQuantizeOnData onData = {
        256, {1, 1}, {0.0f}, {2.55f}, {0.f}, {255.f}, ov::element::u8
    };

    auto decomposed_fq = [](const ov::Output<ov::Node>& input,
                            const ov::element::Type& out_precision,
                            float il, float ih, float scale) -> std::shared_ptr<ov::Node> {
        auto input_low = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {il});
        auto input_high = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {ih});
        auto output_scale = ov::op::v0::Constant::create(ov::element::f32, {1, 1}, {scale});
        auto max_node = std::make_shared<ov::op::v1::Maximum>(input, input_low);
        auto min_node = std::make_shared<ov::op::v1::Minimum>(max_node, input_high);
        return std::make_shared<ov::op::v1::Multiply>(min_node, output_scale);
    };

    std::shared_ptr<ov::Node> current = sub_A;
    current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::f32);
    for (size_t mm_count = 0; mm_count < num_input_nodes; ++mm_count) {
        current = decomposed_fq(current, ov::element::u8, onData.inputLowValues[0], onData.inputHighValues[0], 0.00346764503f);
        current = std::make_shared<ov::snippets::op::ConvertSaturation>(current, ov::element::u8);

        current = std::make_shared<op::TypeRelaxed<ov::op::v0::MatMul>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(sub_trans_zeros0, ov::element::f32).get(),
            false,
            true);

        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
            std::vector<ov::element::Type>{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(sub_zeros2, ov::element::f32).get());
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122f + i});
            current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
                std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
                std::vector<ov::element::Type>{ov::element::f32},
                ov::op::TemporaryReplaceOutputType(current, ov::element::f32).get(),
                ov::op::TemporaryReplaceOutputType(
                    sub_hidden_vec[i], ov::element::f32).get());
        }
    }

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
