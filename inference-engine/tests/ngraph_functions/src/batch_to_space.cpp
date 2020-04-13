// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"


namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeBatchToSpace(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &cropsBegin,
                                               const std::vector<size_t> &cropsEnd) {
    ngraph::Shape constShape = {in.get_shape().size()};
    auto blockShapeNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape,
                                                                     blockShape.data());
    auto cropsBeginNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape,
                                                                     cropsBegin.data());
    auto cropsEndNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape, cropsEnd.data());

    auto btsNode = std::make_shared<ngraph::opset2::BatchToSpace>(in, blockShapeNode, cropsBeginNode, cropsEndNode);
    return btsNode;
}

}  // namespace builder
}  // namespace ngraph