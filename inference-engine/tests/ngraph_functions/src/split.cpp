// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeSplit(const ngraph::Output<Node> &in,
                                        const element::Type &type,
                                        size_t numSplits,
                                        int64_t axis) {
    auto splitAxisOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                  std::vector<int64_t>{axis});
    auto splitNode = std::make_shared<ngraph::opset1::Split>(in, splitAxisOp, numSplits);
    return splitNode;
}
}  // namespace builder
}  // namespace ngraph
