// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeSpaceToBatch(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &padsBegin,
                                               const std::vector<size_t> &padsEnd) {
    ngraph::Shape constShape = {in.get_shape().size()};
    auto blockShapeNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape,
                                                                     blockShape.data());
    auto padsBeginNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape,
                                                                    padsBegin.data());
    auto padsEndNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, constShape, padsEnd.data());
    auto stbNode = std::make_shared<ngraph::opset2::SpaceToBatch>(in, blockShapeNode, padsBeginNode, padsEndNode);
    return stbNode;
}

}  // namespace builder
}  // namespace ngraph