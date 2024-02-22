// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/bitwise_and.hpp"

#include "openvino/op/bitwise_and.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector bitwise_and(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    OPENVINO_ASSERT(inputs.size() == 2);
    return {std::make_shared<v13::BitwiseAnd>(inputs[0], inputs[1])};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
