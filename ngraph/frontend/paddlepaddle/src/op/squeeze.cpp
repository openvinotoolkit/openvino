// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "squeeze.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs squeeze (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    std::vector<int32_t> axes;
    if (node.has_attribute<std::vector<int32_t>>("axes")) {
        axes = node.get_attribute<std::vector<int32_t>>("axes");
    }

    auto axesNode = ngraph::opset6::Constant::create(ngraph::element::i32, {axes.size()}, axes);
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Squeeze>(data, axesNode)}, {"Out"});
}

} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph