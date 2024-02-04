// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/shape.hpp"

#include "openvino/op/shape_of.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector shape(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ng_inputs().at(0);
    return {std::make_shared<v3::ShapeOf>(data)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
OPENVINO_SUPPRESS_DEPRECATED_END
