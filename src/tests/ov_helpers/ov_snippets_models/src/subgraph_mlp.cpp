// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_mlp.hpp"

#include <snippets/op/subgraph.hpp>

#include "openvino/opsets/opset15.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> MLPSeqFunction::initOriginal() const {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, input_shapes[0]);
    auto B =
        std::make_shared<ov::op::v0::Constant>(ov::element::i8, input_shapes[0].to_shape(), std::vector<float>{0.1122});
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                      input_shapes[0].to_shape(),
                                                      std::vector<float>{0.1122});

    ov::builder::subgraph::FakeQuantizeOnData onData =
        {256, {1, 1}, {-1.28f}, {1.27f}, {0.f}, {255.f}, ov::element::u8};
    std::shared_ptr<Node> current = A;

    for (size_t mm_count = 0; mm_count < num_layers; ++mm_count) {
        current = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(current, precisions[0], onData);

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
    auto A = std::make_shared<ov::op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto B = std::make_shared<ov::op::v0::Constant>(ov::element::u8, input_shapes[0].to_shape(), std::vector<float>{0.1122});
    auto add = std::make_shared<ov::op::v0::Constant>(ov::element::u8, input_shapes[0].to_shape(), std::vector<uint8_t>{1});
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{A});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
