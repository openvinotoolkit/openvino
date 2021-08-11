// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/op/util/op_annotations.hpp>
#include <ngraph/opsets/opset4.hpp>

namespace ov {
namespace op {
namespace util {

std::shared_ptr<ov::Node> node_to_get_shape_value_of_indices_from_shape_node(const std::shared_ptr<ov::Node>& shape_node,
                                                                                 const std::vector<size_t>& indices) {
    return std::make_shared<ov::opset4::Gather>(shape_node,
                                                    ov::opset4::Constant::create(ov::element::i64, {indices.size()}, indices),
                                                    ov::opset4::Constant::create(ov::element::i64, {}, {0}));
}

std::shared_ptr<ov::Node> node_to_get_shape_value_of_indices_from_shape_source(const ov::Output<ov::Node>& shape_source,
                                                                                   const std::vector<size_t>& indices) {
    const auto & shape_node = std::make_shared<ov::opset4::ShapeOf>(shape_source);
    return node_to_get_shape_value_of_indices_from_shape_node(shape_node, indices);
}

}  // namespace util
}  // namespace op
}  // namespace ov
