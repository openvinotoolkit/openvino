// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_softmax.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> SoftmaxFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(data, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data});
}

std::shared_ptr<ov::Model> AddSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::op::v1::Add>(data0, data1);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(add, axis);
    return std::make_shared<ov::Model>(NodeVector{softmax}, ParameterVector{data0, data1});
}

std::shared_ptr<ov::Model> TransposeSoftmaxFunction::initOriginal() const {
    const auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    const auto sinh0 = std::make_shared<ov::op::v0::Sinh>(transpose0Param);
    const auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, ov::Shape{m_order.size()}, m_order);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(sinh0, transpose0Const);
    const auto softMax = std::make_shared<ngraph::opset8::Softmax>(transpose2, m_axis);
    return std::make_shared<ov::Model>(ov::NodeVector{softMax}, ov::ParameterVector {transpose0Param}, "softmax_transpose");
}

std::shared_ptr<ov::Model> TransposeSoftmaxEltwiseFunction::initOriginal() const {
    const auto transpose0Param = std::make_shared<ngraph::opset1::Parameter>(precision, input_shapes[0]);
    const auto sinh0 = std::make_shared<ov::op::v0::Sinh>(transpose0Param);
    const auto transpose0Const = ngraph::builder::makeConstant(ngraph::element::i64, ov::Shape{m_order.size()},
                                                               m_order);
    const auto transpose2 = std::make_shared<ov::op::v1::Transpose>(sinh0, transpose0Const);
    const auto mulConst = ngraph::builder::makeConstant(ngraph::element::f32, transpose2->get_shape(),
                                                        std::vector<float>{}, true);
    const auto mul = std::make_shared<ngraph::opset1::Multiply>(transpose2, mulConst);
    const auto softMax = std::make_shared<ngraph::opset8::Softmax>(mul, m_axis);
    const auto hswish = std::make_shared<ngraph::opset6::HSwish>(softMax);
    return std::make_shared<ov::Model>(ov::NodeVector{hswish}, ov::ParameterVector{transpose0Param},
                                       "softmax_transpose");
}

}  // namespace snippets
}  // namespace test
}  // namespace ov