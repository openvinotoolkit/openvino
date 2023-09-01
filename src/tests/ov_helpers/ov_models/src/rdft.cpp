// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

namespace {
    template <typename ...Args>
    std::shared_ptr<ov::Node> CallDftCtorWithArgs(const ov::helpers::DFTOpType opType, Args&&... args) {
        switch (opType) {
            case ov::helpers::DFTOpType::FORWARD:
                return std::make_shared<ov::op::v9::RDFT>(std::forward<Args>(args)...);
            case ov::helpers::DFTOpType::INVERSE:
                return std::make_shared<ov::op::v9::IRDFT>(std::forward<Args>(args)...);
            default:
                throw std::logic_error("Unsupported operation type");
        }
    }
} // namespace

std::shared_ptr<ov::Node> makeRDFT(const ov::Output<Node> &dataNode,
                                      const std::vector<int64_t> &axes,
                                      const std::vector<int64_t> &signalSize,
                                      const ov::helpers::DFTOpType opType) {
    auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{axes.size()}, axes)->output(0);

    if (!signalSize.empty()) {
        auto signalSizeNode = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{signalSize.size()}, signalSize)->output(0);
        return CallDftCtorWithArgs(opType, dataNode, axesNode, signalSizeNode);
    }
    return CallDftCtorWithArgs(opType, dataNode, axesNode);
}
} // namespace builder
} // namespace ov
