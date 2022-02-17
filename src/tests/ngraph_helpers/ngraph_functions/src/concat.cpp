// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeConcat(const std::vector<ngraph::Output<Node>>& in, const int& axis) {
    return std::make_shared<ngraph::opset4::Concat>(in, axis);
}

}  // namespace builder
}  // namespace ngraph