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
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, input_shapes[0]);
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      input_shapes[0].to_shape(),
                                                      std::vector<float>{0.1122});

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ov::element::u8};
    std::shared_ptr<Node> current = A;

    for (size_t mm_count = 0; mm_count < num_layers; ++mm_count) {
        current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, precisions[0], onData);
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
        for (size_t i = 0; i < 2; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                   ov::Shape{64},
                                                                   std::vector<float>{0.1122f + i});
            current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{element::f32},
                ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(constant, element::f32).get());
        }
    }
    current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, precisions[0], onData);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);
    ov::builder::subgraph::FakeQuantizeOnData onData1 =
        {256, {1, 1}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ov::element::i8};
    auto dq_A = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(softmax, precisions[0], onData1);

    auto result = std::make_shared<ov::op::v0::Result>(dq_A);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A});
}

std::shared_ptr<ov::Model> MLPSeqFunction::initReference() const {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, input_shapes[0]);
    auto B0 = std::make_shared<ov::op::v0::Constant>(ov::element::i8, input_shapes[0].to_shape(), std::vector<float>{0.1122f + 0});
    auto B1 = std::make_shared<ov::op::v0::Constant>(ov::element::i8, input_shapes[0].to_shape(), std::vector<float>{0.1122f + 1});
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::i8, input_shapes[0].to_shape(), std::vector<uint8_t>{1});
    auto trans_zeros0 = std::make_shared<ov::op::v1::Transpose>(
        B0, ov::op::v0::Constant::create(ov::element::i8, {input_shapes[0].get_shape().size()}, std::vector<int64_t>{1, 0}));
    auto zeros2 = std::make_shared<ov::op::v0::Constant>(precisions[0], input_shapes[0].to_shape(), std::vector<float>{0.1122});
    auto zeros64 = std::make_shared<ov::op::v0::Constant>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122});
    auto zeros64_2 = std::make_shared<ov::op::v0::Constant>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122 + 1});
    auto zeros64_3 = std::make_shared<ov::op::v0::Constant>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122});
    auto zeros64_4 = std::make_shared<ov::op::v0::Constant>(
        precisions[0], ov::Shape{input_shapes[0].to_shape()[0]}, std::vector<float>{0.1122 + 1});
    auto trans_zeros1 = std::make_shared<ov::op::v1::Transpose>(
        B1, ov::op::v0::Constant::create(ov::element::i8, {input_shapes[0].get_shape().size()}, std::vector<int64_t>{1, 0}));

    auto sub_A = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto sub_trans_zeros0 = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto sub_zeros2 = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto sub_zeros64 = std::make_shared<ov::op::v0::Parameter>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_2 = std::make_shared<ov::op::v0::Parameter>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_3 = std::make_shared<ov::op::v0::Parameter>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_zeros64_4 = std::make_shared<ov::op::v0::Parameter>(precisions[0], ov::Shape{input_shapes[0].to_shape()[0]});
    auto sub_trans_zeros1 = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    ov::ParameterVector subgraph_params = {
        sub_A, sub_trans_zeros0, sub_zeros2, sub_zeros64, sub_zeros64_2, sub_trans_zeros1, sub_zeros2, sub_zeros64_3, sub_zeros64_4};
    ov::NodeVector subgraph_nodes = {A, trans_zeros0, zeros2, zeros64, zeros64_2, trans_zeros1, zeros2, zeros64_3, zeros64_4};

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ov::element::u8};
    std::shared_ptr<Node> current = subgraph_params[0];

    for (size_t mm_count = 0; mm_count < num_layers; ++mm_count) {
        current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, precisions[0], onData);

        current = std::make_shared<op::TypeRelaxed<ov::op::v0::MatMul>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(mm_count == 0 ? B0 : B1, element::f32).get(),
            false,
            true);

        current = std::make_shared<op::TypeRelaxed<ov::op::v1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(add, element::f32).get());
        for (size_t i = 0; i < 2; ++i) {
            auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                   ov::Shape{64},
                                                                   std::vector<float>{0.1122f + i});
            current = std::make_shared<op::TypeRelaxed<ov::op::v1::Add>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{element::f32},
                ov::op::TemporaryReplaceOutputType(current, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(constant, element::f32).get());
        }
    }
    current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, precisions[0], onData);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(current, 1);
    ov::builder::subgraph::FakeQuantizeOnData onData1 =
        {256, {1, 1}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ov::element::i8};
    auto dq_A = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(softmax, precisions[0], onData1);
    auto result_subgraph = std::make_shared<ov::op::v0::Result>(dq_A);

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(subgraph_nodes, std::make_shared<Model>(ResultVector{result_subgraph}, subgraph_params));
    auto result = std::make_shared<ov::op::v0::Result>(subgraph);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
