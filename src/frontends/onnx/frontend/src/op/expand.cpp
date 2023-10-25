// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/expand.hpp"

#include <memory>

#include "default_opset.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector expand(const Node& node) {
    const Output<ngraph::Node> data{node.get_ng_inputs().at(0)};
    const Output<ngraph::Node> shape{node.get_ng_inputs().at(1)};

    if (common::is_failsafe_node(shape.get_node_shared_ptr())) {
        // in case the "shape" input is connected to a failsafe node created in place of an invalid initializer
        // the target shape should be ignored and this Expand operation should not modify its input tensor
        // the Broadcast created below should be eliminated later on by an appropriate optimization pass
        const auto identity_broadcast = default_opset::Constant::create(element::i64, Shape{1}, {1});
        return {std::make_shared<default_opset::Broadcast>(data,
                                                           identity_broadcast,
                                                           ngraph::op::BroadcastType::BIDIRECTIONAL)};
    } else {
        return {std::make_shared<default_opset::Broadcast>(data, shape, ngraph::op::BroadcastType::BIDIRECTIONAL)};
    }
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
