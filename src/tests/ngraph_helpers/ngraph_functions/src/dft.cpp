// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

namespace {
template <typename... Args>
std::shared_ptr<Node> CallDftCtorWithArgs(const helpers::DFTOpType opType,
                                          const helpers::DFTOpMode opMode,
                                          Args&&... args) {
    if (opType == helpers::DFTOpType::FORWARD && opMode == helpers::DFTOpMode::COMPLEX) {
        return std::make_shared<op::v7::DFT>(std::forward<Args>(args)...);
    } else if (opType == helpers::DFTOpType::INVERSE && opMode == helpers::DFTOpMode::COMPLEX) {
        return std::make_shared<op::v7::IDFT>(std::forward<Args>(args)...);
    } else if (opType == helpers::DFTOpType::FORWARD && opMode == helpers::DFTOpMode::REAL) {
        return std::make_shared<op::v9::RDFT>(std::forward<Args>(args)...);
    } else if (opType == helpers::DFTOpType::INVERSE && opMode == helpers::DFTOpMode::REAL) {
        return std::make_shared<op::v9::IRDFT>(std::forward<Args>(args)...);
    } else {
        throw std::logic_error("Unsupported DFT operation.");
    }
}
}  // namespace

std::shared_ptr<Node> makeDFT(const Output<Node>& dataNode,
                              const std::vector<int64_t>& axes,
                              const std::vector<int64_t>& signalSize,
                              const helpers::DFTOpType opType,
                              const helpers::DFTOpMode opMode) {
    auto axesNode = std::make_shared<op::Constant>(element::Type_t::i64, Shape{axes.size()}, axes)->output(0);

    if (!signalSize.empty()) {
        auto signalSizeNode =
            std::make_shared<op::Constant>(element::Type_t::i64, Shape{signalSize.size()}, signalSize)->output(0);
        return CallDftCtorWithArgs(opType, opMode, dataNode, axesNode, signalSizeNode);
    }
    return CallDftCtorWithArgs(opType, opMode, dataNode, axesNode);
}
}  // namespace builder
}  // namespace ngraph
