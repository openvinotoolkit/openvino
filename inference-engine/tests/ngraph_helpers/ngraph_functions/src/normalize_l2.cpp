// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeNormalizeL2(const ngraph::Output<Node>& data,
                                              const std::vector<int64_t>& axes,
                                              float eps,
                                              ngraph::op::EpsMode epsMode) {
    auto normAxes = std::make_shared<ngraph::opset4::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{axes.size()}, axes);
    return std::make_shared<ngraph::opset4::NormalizeL2>(data, normAxes, eps, epsMode);
}
}  // namespace builder
}  // namespace ngraph
