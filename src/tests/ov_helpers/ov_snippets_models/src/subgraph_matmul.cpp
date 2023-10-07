// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_matmul.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ov_models/builders.hpp"
#include "ov_ops/type_relaxed.hpp"


namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    std::shared_ptr<Node> matmul;
    if (precisions[1] == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                std::vector<element::Type>{element::f32, element::f32},
                std::vector<element::Type>{ element::f32 },
                ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    } else {
        matmul = std::make_shared<op::v0::MatMul>(data0, data1);
    }
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> MatMulFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto indata0 = std::make_shared<op::v0::Parameter>(precisions[0], data0->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(precisions[1], data1->get_output_partial_shape(0));
    std::shared_ptr<Node> matmul;
    if (precisions[1] == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ov::op::TemporaryReplaceOutputType(indata0, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(indata1, element::f32).get());
    } else {
        matmul = std::make_shared<op::v0::MatMul>(indata0, indata1);
    }
    const auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1},
                                                                std::make_shared<ov::Model>(NodeVector{matmul},
                                                                                            ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> FQMatMulFunction::initOriginal() const {
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto ih = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{34.7436294});
    auto il = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{-35.0172004});
    auto oh = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{34.7436294});
    auto ol = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{-35.0172004});
    auto fq = std::make_shared<op::v0::FakeQuantize>(data0, il, ih, ol, oh, 256);
    std::shared_ptr<ov::Node> in0 = fq;
    if (pos == 0) {
        in0 = std::make_shared<op::v1::Transpose>(in0, const_order);
    }
    auto constant = ngraph::builder::makeConstant(ov::element::i8, const_shape.get_shape(), std::vector<int8_t>{}, true);
    auto convert = std::make_shared<op::v0::Convert>(constant, ov::element::f32);
    auto deq_mul = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.00499185826});
    auto mul = std::make_shared<op::v1::Multiply>(convert, deq_mul);
    std::shared_ptr<ov::Node> in1 = mul;
    if (pos == 1) {
        in1 = std::make_shared<op::v1::Transpose>(in1, const_order);
    }
    auto matmul = std::make_shared<op::v0::MatMul>(in0, in1);
    std::shared_ptr<ov::Node> out = matmul;
    if (pos == 2) {
        out = std::make_shared<op::v1::Transpose>(out, const_order);
    }
    return std::make_shared<ov::Model>(NodeVector{out}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> MatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    std::shared_ptr<Node> matmul;
    if (precisions[1]  == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    } else {
        matmul = std::make_shared<op::v0::MatMul>(data0, data1);
    }
    auto bias = std::make_shared<op::v1::Add>(matmul, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> MatMulBiasQuantizedFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                  std::vector<element::Type>{element::f32, element::f32},
                  std::vector<element::Type>{ element::f32 },
                  ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                  ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto fq2 = ngraph::builder::makeFakeQuantize(matmul, ov::element::f32, 256, {1}, {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    auto bias = std::make_shared<op::v1::Add>(fq2, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> MatMulsQuantizedFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto matmul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                   std::vector<element::Type>{element::f32, element::f32},
                   std::vector<element::Type>{ element::f32 },
                   ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                   ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto fq0 = ngraph::builder::makeFakeQuantize(matmul0, ov::element::f32, 256, {1}, {0}, {0.820726}, {0}, {0.820726});
    auto fq2 = ngraph::builder::makeFakeQuantize(data2, ov::element::f32, 256, {1}, {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    auto new_shape = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{4},
                                                            std::vector<uint64_t>{1, 1, input_shapes[2].get_shape()[0], input_shapes[2].get_shape()[1]});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(fq2, new_shape, false);
    auto matmul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                   std::vector<element::Type>{element::f32, element::f32},
                   std::vector<element::Type>{ element::f32 },
                   ov::op::TemporaryReplaceOutputType(fq0, element::f32).get(),
                   ov::op::TemporaryReplaceOutputType(reshape, element::f32).get());
     auto fq3 = ngraph::builder::makeFakeQuantize(matmul1, ov::element::f32, 256, {1}, {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    return std::make_shared<ov::Model>(NodeVector{fq3}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> Transpose0213MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    std::shared_ptr<Node> result;
    switch (transpose_position) {
        case 0: {
            auto transpose = std::make_shared<op::v1::Transpose>(data0, const_order);
            if (precisions[1] == ov::element::i8) {
                result = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                         std::vector<element::Type>{element::f32, element::f32},
                         std::vector<element::Type>{ element::f32 },
                         ov::op::TemporaryReplaceOutputType(transpose, element::f32).get(),
                         ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
            } else {
                result = std::make_shared<op::v0::MatMul>(transpose, data1);
            }
            break;
        } case 1: {
            auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
            if (precisions[1] == ov::element::i8) {
                result = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                         std::vector<element::Type>{element::f32, element::f32},
                         std::vector<element::Type>{ element::f32 },
                         ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                         ov::op::TemporaryReplaceOutputType(transpose, element::f32).get());
            } else {
                result = std::make_shared<op::v0::MatMul>(data0, transpose);
            }
            break;
        } case 2: {
            std::shared_ptr<ov::Node> matmul;
            if (precisions[1] == ov::element::i8) {
                matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                         std::vector<element::Type>{element::f32, element::f32},
                         std::vector<element::Type>{ element::f32 },
                         ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                         ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
            } else {
                matmul = std::make_shared<op::v0::MatMul>(data0, data1);
            }
            result = std::make_shared<op::v1::Transpose>(matmul, const_order);
            break;
        }
    }
    return std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TransposeMatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, transpose);
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> TransposeMatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, transpose);
    auto bias = std::make_shared<op::v1::Add>(matmul, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> TransposeMulMatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<op::v0::Parameter>(precision, input_shapes[3]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto mul = std::make_shared<op::v1::Multiply>(transpose, data2);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, mul);
    auto bias = std::make_shared<op::v1::Add>(matmul, data3);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2, data3});
}
std::shared_ptr<ov::Model> MatMulsQuantizedSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto matmul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                   std::vector<element::Type>{element::f32, element::f32},
                   std::vector<element::Type>{ element::f32 },
                   ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                   ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul0, -1);
    auto fq0 = ngraph::builder::makeFakeQuantize(softmax, ov::element::f32, 256, {1}, {0}, {0.820726}, {0}, {0.820726});
    auto fq2 = ngraph::builder::makeFakeQuantize(data2, ov::element::f32, 256, {1}, {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    auto new_shape = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{4},
                                                            std::vector<uint64_t>{1, 1, input_shapes[2].get_shape()[0], input_shapes[2].get_shape()[1]});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(fq2, new_shape, false);
    auto matmul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                   std::vector<element::Type>{element::f32, element::f32},
                   std::vector<element::Type>{ element::f32 },
                   ov::op::TemporaryReplaceOutputType(fq0, element::f32).get(),
                   ov::op::TemporaryReplaceOutputType(reshape, element::f32).get());
     auto fq3 = ngraph::builder::makeFakeQuantize(matmul1, ov::element::f32, 256, {1}, {-35.0172004}, {34.7436294}, {-35.0172004}, {34.7436294});
    return std::make_shared<ov::Model>(NodeVector{fq3}, ParameterVector{data0, data1, data2});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
