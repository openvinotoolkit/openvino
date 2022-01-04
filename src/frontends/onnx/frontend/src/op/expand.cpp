// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/expand.hpp"

#include <memory>

#include "default_opset.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector expand(const Node& node) {
    const Output<ov::Node> data{node.get_ng_inputs().at(0)};
    const Output<ov::Node> shape{node.get_ng_inputs().at(1)};

    return {std::make_shared<default_opset::Broadcast>(data, shape, ov::op::BroadcastType::BIDIRECTIONAL)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
