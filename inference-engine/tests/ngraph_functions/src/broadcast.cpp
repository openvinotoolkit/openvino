// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeBroadcast(const Output<Node> &in,
                                    const op::BroadcastModeSpec &mode,
                                    const std::vector<size_t> &targetShape,
                                    const std::vector<size_t> &axesMapping) {
    auto target = std::make_shared<ngraph::opset1::Constant>(element::i64, std::vector<size_t>{targetShape.size()}, targetShape);
    if (mode.m_type == op::BroadcastType::EXPLICIT) {
        auto axes = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, std::vector<size_t>{axesMapping.size()}, axesMapping);
        return std::make_shared<ngraph::opset3::Broadcast>(in, target, axes, mode);
    } else {
        return std::make_shared<ngraph::opset3::Broadcast>(in, target, mode);
    }
}

}  // namespace builder
}  // namespace ngraph