// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_matmul.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> MatMulSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto matmul = std::make_shared<op::v0::MatMul>(sinh0, sinh1);
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> MatMulSinhFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, sinh0->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, sinh1->get_output_partial_shape(0));
    auto matmul = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{sinh0, sinh1},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v0::MatMul>(indata0, indata1)},
                                                                      ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> MatMulBiasSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto matmul = std::make_shared<op::v0::MatMul>(sinh0, sinh1);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto sinh2 = std::make_shared<ov::op::v0::Sinh>(data2);
    auto bias = std::make_shared<op::v1::Add>(matmul, sinh2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> Transpose0213MatMulSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data0_guarded = insert_guard ? std::make_shared<ov::op::v0::Sinh>(data0)->output(0) : data0->output(0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data1_guarded = insert_guard ? std::make_shared<ov::op::v0::Sinh>(data1)->output(0) : data1->output(0);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    std::shared_ptr<Node> result;
    switch (transpose_position) {
        case 0: {
            auto transpose = std::make_shared<op::v1::Transpose>(data0_guarded, const_order);
            result = std::make_shared<op::v0::MatMul>(transpose, data1_guarded);
            break;
        } case 1: {
            auto transpose = std::make_shared<op::v1::Transpose>(data1_guarded, const_order);
            result = std::make_shared<op::v0::MatMul>(data0_guarded, transpose);
            break;
        } case 2: {
            auto matmul = std::make_shared<op::v0::MatMul>(data0_guarded, data1_guarded);
            result = std::make_shared<op::v1::Transpose>(matmul, const_order);
            break;
        }
    }
    return std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TransposeMatMulSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(sinh1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(sinh0, transpose);
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> TransposeMatMulBiasSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto sinh2 = std::make_shared<ov::op::v0::Sinh>(data2);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(sinh1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(sinh0, transpose);
    auto bias = std::make_shared<op::v1::Add>(matmul, sinh2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> TransposeMulMatMulBiasSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<op::v0::Parameter>(precision, input_shapes[3]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto sinh2 = std::make_shared<ov::op::v0::Sinh>(data2);
    auto sinh3 = std::make_shared<ov::op::v0::Sinh>(data3);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(sinh1, const_order);
    auto mul = std::make_shared<op::v1::Multiply>(transpose, sinh2);
    auto matmul = std::make_shared<op::v0::MatMul>(sinh0, mul);
    auto bias = std::make_shared<op::v1::Add>(matmul, sinh3);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2, data3});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov