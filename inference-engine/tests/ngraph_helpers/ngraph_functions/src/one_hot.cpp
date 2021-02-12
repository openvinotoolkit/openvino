// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {


std::shared_ptr<ngraph::Node>  makeOneHot(const ngraph::Output<Node>& indices,
                                          const element::Type& depth_type,
                                          const int64_t& depth_val,
                                          const element::Type& set_type,
                                          const float& on_val,
                                          const float& off_val,
                                          const int64_t& axis) {
    auto depth_const = std::make_shared<ngraph::op::Constant>(depth_type, ngraph::Shape{ }, depth_val);
    auto on_value_const = std::make_shared<ngraph::op::Constant>(set_type, ngraph::Shape{ }, on_val);
    auto off_value_const = std::make_shared<ngraph::op::Constant>(set_type, ngraph::Shape{ }, off_val);
    return std::make_shared<ngraph::opset5::OneHot>(indices, depth_const, on_value_const, off_value_const, axis);
}
}  // namespace builder
}  // namespace ngraph
