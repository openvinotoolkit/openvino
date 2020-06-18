// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeSqueeze(const ngraph::Output<Node> &in,
                                          const element::Type &type,
                                          const std::vector<int> &squeeze_indices,
                                          ngraph::helpers::SqueezeOpType opType,
                                          bool isScalar) {
    auto squeeze_node = std::make_shared<ngraph::opset1::Constant>(type,
            ngraph::Shape{isScalar ? 0 :squeeze_indices.size()},
            squeeze_indices);
    switch (opType) {
        case ngraph::helpers::SqueezeOpType::SQUEEZE:
            return std::make_shared<ngraph::opset1::Squeeze>(in, squeeze_node);
        case ngraph::helpers::SqueezeOpType::UNSQUEEZE:
            return std::make_shared<ngraph::opset1::Unsqueeze>(in, squeeze_node);
        default:
            std::logic_error("Unsupported operation type");
    }
}
} // namespace builder
} // namespace ngraph
