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
std::shared_ptr<ov::Model> Transpose0213MatMulSinhFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto sinh0 = std::make_shared<ov::op::v0::Sinh>(data0);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto sinh1 = std::make_shared<ov::op::v0::Sinh>(data1);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    std::shared_ptr<Node> result;
    switch (transpose_position) {
        case 0: {
            auto transpose = std::make_shared<op::v1::Transpose>(sinh0, const_order);
            result = std::make_shared<op::v0::MatMul>(transpose, sinh1);
            break;
        } case 1: {
            auto transpose = std::make_shared<op::v1::Transpose>(sinh1, const_order);
            result = std::make_shared<op::v0::MatMul>(sinh0, transpose);
            break;
        } case 2: {
            auto matmul = std::make_shared<op::v0::MatMul>(sinh0, sinh1);
            result = std::make_shared<op::v1::Transpose>(matmul, const_order);
            break;
        }
    }
    return std::make_shared<ov::Model>(NodeVector{result}, ParameterVector{data0, data1});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov