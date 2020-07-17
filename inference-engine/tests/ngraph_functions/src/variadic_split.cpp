// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
        std::shared_ptr<ngraph::Node> makeVariadicSplit(const ngraph::Output<Node> &in,
                                                        const element::Type &type,
                                                        const std::vector<size_t> numSplits,
                                                        size_t axis) {
            auto splitAxisOp = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                          std::vector<uint64_t>{axis});
            auto param = ngraph::builder::makeParams(type, {numSplits});
            auto numSplit = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(param));
//            auto numSplit = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
//                                                                       numSplits);
            auto VariadicSplitNode = std::make_shared<ngraph::opset3::VariadicSplit>(in, splitAxisOp, numSplit[0]);
            return VariadicSplitNode;
        }
}  // namespace builder
}  // namespace ngraph
