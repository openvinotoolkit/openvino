// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<ov::Node> makeSelect(std::vector<ov::Output<Node>> &in,
                                         const ov::op::AutoBroadcastSpec& auto_broadcast) {
    auto selectNode = std::make_shared<ov::opset1::Select>(in[0], in[1], in[2], auto_broadcast);
    return selectNode;
}

}  // namespace builder
}  // namespace ov
