// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_matmul.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, data1);
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> MatMulFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(precision, data1->get_output_partial_shape(0));
    auto matmul = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{data0, data1},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v0::MatMul>(indata0, indata1)},
                                                                      ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> MatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, data1);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto bias = std::make_shared<op::v1::Add>(matmul, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> Transpose0213MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    std::shared_ptr<Node> result;
    switch (transpose_position) {
        case 0: {
            auto transpose = std::make_shared<op::v1::Transpose>(data0, const_order);
            result = std::make_shared<op::v0::MatMul>(transpose, data1);
            break;
        } case 1: {
            auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
            result = std::make_shared<op::v0::MatMul>(data0, transpose);
            break;
        } case 2: {
            auto matmul = std::make_shared<op::v0::MatMul>(data0, data1);
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

}  // namespace snippets
}  // namespace test
}  // namespace ov