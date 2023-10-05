// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/dft.hpp"

#include <memory>
#include <vector>

#include "openvino/op/idft.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

namespace {
template <typename... Args>
std::shared_ptr<ov::Node> CallDftCtorWithArgs(const ov::test::utils::DFTOpType opType, Args&&... args) {
    switch (opType) {
    case ov::test::utils::DFTOpType::FORWARD:
        return std::make_shared<ov::op::v7::DFT>(std::forward<Args>(args)...);
    case ov::test::utils::DFTOpType::INVERSE:
        return std::make_shared<ov::op::v7::IDFT>(std::forward<Args>(args)...);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}
}  // namespace

std::shared_ptr<ov::Node> makeDFT(const ov::Output<Node>& dataNode,
                                  const std::vector<int64_t>& axes,
                                  const std::vector<int64_t>& signalSize,
                                  const ov::test::utils::DFTOpType opType) {
    auto axesNode =
        std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{axes.size()}, axes)->output(0);

    if (!signalSize.empty()) {
        auto signalSizeNode =
            std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{signalSize.size()}, signalSize)
                ->output(0);
        return CallDftCtorWithArgs(opType, dataNode, axesNode, signalSizeNode);
    }
    return CallDftCtorWithArgs(opType, dataNode, axesNode);
}
}  // namespace builder
}  // namespace ngraph
