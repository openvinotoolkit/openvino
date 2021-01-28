// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {


std::shared_ptr<ngraph::Node>  makeOneHot(const ngraph::Output<Node>& indices,
                                          const std::pair<ngraph::element::Type, int64_t>& depth_type_val,
                                          const std::pair<ngraph::element::Type, float>& on_type_val,
                                          const std::pair<ngraph::element::Type, float>& off_type_val,
                                          const int64_t& axis) {
    auto depth_const = std::make_shared<ngraph::op::Constant>(depth_type_val.first, ngraph::Shape{ }, depth_type_val.second);
    auto on_value_const = std::make_shared<ngraph::op::Constant>(on_type_val.first, ngraph::Shape{ }, on_type_val.second);
    auto off_value_const = std::make_shared<ngraph::op::Constant>(off_type_val.first, ngraph::Shape{ }, off_type_val.second);
    return std::make_shared<ngraph::opset3::OneHot>(indices, depth_const, on_value_const, off_value_const, axis);
}
}  // namespace builder
}  // namespace ngraph
