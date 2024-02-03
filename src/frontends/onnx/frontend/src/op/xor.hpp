// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"
#include "openvino/op/logical_xor.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector logical_xor(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalXor>(node.get_ng_inputs().at(0),
                                                     node.get_ng_inputs().at(1),
                                                     ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY))};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
OPENVINO_SUPPRESS_DEPRECATED_END
