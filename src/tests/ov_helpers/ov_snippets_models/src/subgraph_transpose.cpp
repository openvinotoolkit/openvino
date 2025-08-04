// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_transpose.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {
std::shared_ptr<ov::Model> TransposeFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {order.size()}, order);
    auto transpose = std::make_shared<op::v1::Transpose>(data, const_order);
    return std::make_shared<ov::Model>(OutputVector{transpose}, ParameterVector{data});
}
std::shared_ptr<ov::Model> TransposeFunction::initReference() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {order.size()}, order);
    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(const_order->get_output_element_type(0),
                                                       const_order->get_output_partial_shape(0));
    auto transpose = std::make_shared<ov::snippets::op::Subgraph>(
        NodeVector{data, const_order},
        std::make_shared<ov::Model>(OutputVector{std::make_shared<op::v1::Transpose>(indata0, indata1)},
                                    ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(OutputVector{transpose}, ParameterVector{data});
}
std::shared_ptr<ov::Model> TransposeMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {order.size()}, order);
    auto transpose = std::make_shared<op::v1::Transpose>(data0, const_order);
    auto multiply = std::make_shared<op::v1::Multiply>(transpose, data1);
    return std::make_shared<ov::Model>(OutputVector{multiply}, ParameterVector{data0, data1});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
