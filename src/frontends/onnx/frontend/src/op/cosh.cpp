// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cosh.hpp"

#include "openvino/op/cosh.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector cosh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Cosh>(node.get_ov_inputs().at(0))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
