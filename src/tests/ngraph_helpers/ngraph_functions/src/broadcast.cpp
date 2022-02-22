// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ngraph::Node> makeBroadcast(const ngraph::Output<Node> &in,
                                            const ngraph::Output<Node> &target_shape,
                                            const ngraph::op::BroadcastType& mode,
                                            const ngraph::AxisSet& axisSet) {
    if (mode == ngraph::op::BroadcastType::NONE) {
        auto axisSetConst = ngraph::opset5::Constant::create(ngraph::element::i64, {axisSet.size()}, axisSet.to_vector());
        return std::make_shared<ngraph::opset5::Broadcast>(in,
                                                           target_shape,
                                                           axisSetConst,
                                                           mode);
    } else { // numpy/bidirectional modes
        return std::make_shared<ngraph::opset5::Broadcast>(in,
                                                           target_shape,
                                                           mode);
    }
}
}  // namespace builder
}  // namespace ngraph
