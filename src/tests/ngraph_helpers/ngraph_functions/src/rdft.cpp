// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rdft.hpp"

#include <memory>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/irdft.hpp"

namespace ngraph {
namespace builder {

namespace {
template <typename... Args>
std::shared_ptr<ov::Node> CallDftCtorWithArgs(const ov::test::utils::DFTOpType opType, Args&&... args) {
    switch (opType) {
    case ov::test::utils::DFTOpType::FORWARD:
        return std::make_shared<ov::op::v9::RDFT>(std::forward<Args>(args)...);
    case ov::test::utils::DFTOpType::INVERSE:
        return std::make_shared<ov::op::v9::IRDFT>(std::forward<Args>(args)...);
    default:
        throw std::logic_error("Unsupported operation type");
    }
}
}  // namespace

std::shared_ptr<ov::Node> makeRDFT(const ov::Output<Node>& dataNode,
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
