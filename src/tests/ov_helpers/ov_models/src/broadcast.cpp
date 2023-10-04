// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"

#include <memory>
#include <vector>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {
std::shared_ptr<ov::Node> makeBroadcast(const ov::Output<Node>& in,
                                        const ov::Output<Node>& target_shape,
                                        const ov::op::BroadcastType& mode,
                                        const ov::AxisSet& axisSet) {
    if (mode == ov::op::BroadcastType::NONE) {
        auto axisSetConst = ov::op::v0::Constant::create(ov::element::i64, {axisSet.size()}, axisSet.to_vector());
        return std::make_shared<ov::op::v3::Broadcast>(in, target_shape, axisSetConst, mode);
    } else {  // numpy/bidirectional modes
        return std::make_shared<ov::op::v3::Broadcast>(in, target_shape, mode);
    }
}
}  // namespace builder
}  // namespace ngraph
