// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeSelect(std::vector<ngraph::Output<Node>> &in,
                                         const ngraph::op::AutoBroadcastSpec& auto_broadcast) {
    auto selectNode = std::make_shared<ngraph::opset1::Select>(in[0], in[1], in[2], auto_broadcast);
    return selectNode;
}

}  // namespace builder
}  // namespace ngraph
