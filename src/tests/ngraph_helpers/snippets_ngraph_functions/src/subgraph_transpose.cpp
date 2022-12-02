// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_transpose.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> TransposeSinhFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {order.size()}, order);
    auto sinh = std::make_shared<ov::op::v0::Sinh>(data);
    auto transpose = std::make_shared<op::v1::Transpose>(sinh, const_order);
    return std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{data});
}
std::shared_ptr<ov::Model> TransposeSinhFunction::initReference() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {order.size()}, order);
    auto sinh = std::make_shared<ov::op::v0::Sinh>(data);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, sinh->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(const_order->get_output_element_type(0),
                                                       const_order->get_output_partial_shape(0));
    auto transpose = std::make_shared<ngraph::snippets::op::Subgraph>(NodeVector{sinh, const_order},
                                          std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v1::Transpose>(indata0, indata1)},
                                                                      ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{data});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov