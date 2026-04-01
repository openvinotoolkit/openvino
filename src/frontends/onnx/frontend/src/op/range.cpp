// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/squeeze.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector range(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node, inputs.size() >= 3, "Minimum 3 inputs are required. Got: ", inputs.size());

    ov::Output<ov::Node> start{inputs[0]};
    ov::Output<ov::Node> stop{inputs[1]};
    ov::Output<ov::Node> step{inputs[2]};

    auto axes = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});

    const auto is_singleton_vector = [](const ov::PartialShape& pshape) {
        return pshape.rank().is_static() && pshape.rank().get_length() == 1 && pshape[0].is_static() &&
               pshape[0].get_length() == 1;
    };

    // Squeeze 1D-tensors with single element into scalars for Range inputs
    if (is_singleton_vector(start.get_partial_shape())) {
        start = std::make_shared<v0::Squeeze>(start, axes);
    }

    if (is_singleton_vector(stop.get_partial_shape())) {
        stop = std::make_shared<v0::Squeeze>(stop, axes);
    }

    if (is_singleton_vector(step.get_partial_shape())) {
        step = std::make_shared<v0::Squeeze>(step, axes);
    }

    return {std::make_shared<v4::Range>(start, stop, step, start.get_element_type())};
}

ONNX_OP("Range", OPSET_SINCE(1), ai_onnx::opset_1::range);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
