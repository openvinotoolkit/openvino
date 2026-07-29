// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/concat_from_sequence.hpp"

#include <cstdint>
#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

ov::OutputVector concat_from_sequence(const ov::frontend::onnx::Node& node) {
    constexpr auto input_only = 1;
    common::default_op_checks(node, input_only, input_only);

    const auto axis = node.get_attribute_value<int64_t>("axis", 0);
    const auto new_axis = node.get_attribute_value<int64_t>("new_axis", 0) != 0;

    const auto& inputs = node.get_ov_inputs();
    const auto& sequence_input = inputs.front();

    // Create a ConcatFromSequence helper op that will be resolved by a transformation pass
    return {std::make_shared<ov::frontend::ConcatFromSequence>(sequence_input, axis, new_axis)};
}

ONNX_OP("ConcatFromSequence", OPSET_SINCE(1), ai_onnx::opset_11::concat_from_sequence);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
