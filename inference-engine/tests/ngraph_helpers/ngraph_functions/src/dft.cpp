// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

namespace {
    template <typename ...Args>
    std::shared_ptr<ngraph::Node> CallDftCtorWithArgs(const ngraph::helpers::DFTOpType opType, Args&&... args) {
        switch (opType) {
            case ngraph::helpers::DFTOpType::FORWARD:
                return std::make_shared<ngraph::op::v7::DFT>(std::forward<Args>(args)...);
            case ngraph::helpers::DFTOpType::INVERSE:
                return std::make_shared<ngraph::op::v7::IDFT>(std::forward<Args>(args)...);
            default:
                throw std::logic_error("Unsupported operation type");
        }
    }
} // namespace

std::shared_ptr<ngraph::Node> makeDFT(const ngraph::Output<Node> &dataNode,
                                      const std::vector<int64_t> &axes,
                                      const std::vector<int64_t> &signalSize,
                                      const ngraph::helpers::DFTOpType opType) {
    auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{axes.size()}, axes)->output(0);

    if (!signalSize.empty()) {
        auto signalSizeNode = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{signalSize.size()}, signalSize)->output(0);
        return CallDftCtorWithArgs(opType, dataNode, axesNode, signalSizeNode);
    }
    return CallDftCtorWithArgs(opType, dataNode, axesNode);
}
} // namespace builder
} // namespace ngraph
