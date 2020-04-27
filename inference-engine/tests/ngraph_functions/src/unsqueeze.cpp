// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeUnsqueeze(const ngraph::Output<Node> &in,
                                          const element::Type &type,
                                          const std::vector<size_t> &squeeze_indices) {
    auto squeeze_node = std::make_shared<ngraph::opset1::Constant>(type,
                                                                   ngraph::Shape{squeeze_indices.size()},
                                                                   std::vector<size_t>{squeeze_indices});
    auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(in, squeeze_node);
    return unsqueeze;
}
} // namespace builder
} // namespace ngraph
