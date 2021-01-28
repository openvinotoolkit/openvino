// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {


std::shared_ptr<ngraph::Node>  makeOneHot(const ngraph::Output<Node>& indices,
                                          int64_t depth,
                                          float on_value,
                                          float off_value,
                                          int64_t axis) {
    auto depth_const = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ }, depth);
    //auto on_value_const = std::make_shared<ngraph::op::Constant>(indices.get_element_type(), ngraph::Shape{ }, on_value);
    //auto off_value_const = std::make_shared<ngraph::op::Constant>(indices.get_element_type(), ngraph::Shape{ }, off_value);

    auto on_value_const = std::make_shared<ngraph::op::Constant>(ngraph::element::f16, ngraph::Shape{ }, on_value);
    auto off_value_const = std::make_shared<ngraph::op::Constant>(ngraph::element::f16, ngraph::Shape{ }, off_value);
    return std::make_shared<ngraph::opset3::OneHot>(indices, depth_const, on_value_const, off_value_const, axis);
}
}  // namespace builder
}  // namespace ngraph
