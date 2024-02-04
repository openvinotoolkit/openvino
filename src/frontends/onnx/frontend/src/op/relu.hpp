// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "openvino/op/relu.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector relu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ng_inputs{node.get_ng_inputs()};
    return {std::make_shared<ov::op::v0::Relu>(ng_inputs.at(0))};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
