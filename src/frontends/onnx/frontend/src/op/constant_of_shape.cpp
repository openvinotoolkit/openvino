// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/constant_of_shape.hpp"

#include "core/null_node.hpp"
#include "core/tensor.hpp"
#include "op/constant.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector constant_of_shape(const ov::frontend::onnx::Node& node) {
    ov::Output<ov::Node> constant_value;
    if (node.has_attribute("value")) {
        auto value_tensor = node.get_attribute_value<Tensor>("value");
        constant_value = value_tensor.get_ov_constant();
        constant_value = reshape::interpret_as_scalar(constant_value);
    } else {
        constant_value = v0::Constant::create(ov::element::f32, {}, {0});
    }
    const auto& inputs = node.get_ov_inputs();
    if (inputs.size() == 0 || common::is_failsafe_node(inputs[0].get_node_shared_ptr()) ||
        ov::op::util::is_null(inputs[0])) {
        return {constant_value};
    }
    return {std::make_shared<v3::Broadcast>(constant_value, inputs[0])};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
