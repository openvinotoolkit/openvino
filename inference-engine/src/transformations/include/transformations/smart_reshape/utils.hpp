// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <assert.h>
#include <vector>
#include <limits>

#include <transformations_visibility.hpp>
#include <ngraph/op/util/op_annotations.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>

namespace ngraph {
namespace op {
namespace util {

std::shared_ptr<ngraph::Node> node_to_get_shape_value_of_indices_from_shape_node(const std::shared_ptr<ngraph::Node>& shape_node,
                                                                                 const std::vector<size_t>& indices) {
    return std::make_shared<ngraph::opset4::Gather>(shape_node,
                                                    ngraph::opset4::Constant::create(ngraph::element::i64, {indices.size()}, indices),
                                                    ngraph::opset4::Constant::create(ngraph::element::i64, {}, {0}));
}

std::shared_ptr<ngraph::Node> node_to_get_shape_value_of_indices_from_shape_source(const ngraph::Output<ngraph::Node>& shape_source,
                                                                                   const std::vector<size_t>& indices) {
    const auto & shape_node = std::make_shared<ngraph::opset4::ShapeOf>(shape_source);
    return node_to_get_shape_value_of_indices_from_shape_node(shape_node, indices);
}

}  // namespace util
}  // namespace op
}  // namespace ngraph