// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/range.hpp"

#include "exceptions.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/squeeze.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector range(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node, inputs.size() >= 3, "Maximum 3 inputs are required. Got: ", inputs.size());

    ov::Output<ov::Node> start{inputs[0]};
    ov::Output<ov::Node> limit{inputs[1]};
    ov::Output<ov::Node> delta{inputs[2]};

    auto axes = std::make_shared<v0::Constant>(ov::element::i64, oc::Shape{}, std::vector<int64_t>{0});

    // Check if delta is a tensor with single value
    if (delta.get_shape().size() == 1 && delta.get_shape()[0] == 1) {
        delta = std::make_shared<v0::Squeeze>(delta, axes);
    }

    return {
        std::make_shared<v4::Range>(start, limit, delta, start.get_element_type())
    }
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov