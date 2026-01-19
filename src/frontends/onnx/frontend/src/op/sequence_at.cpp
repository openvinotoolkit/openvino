// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

/// @brief Implements the SequenceAt operator
/// @param node Input ONNX node. Must have two inputs: a sequence and a position.
///             Sequence is represented as a SequenceMark node.
/// @return The tensor at the specified position in the input sequence
ov::OutputVector sequence_at(const ov::frontend::onnx::Node& node) {
    constexpr auto input_sequence_and_position = 2;

    common::default_op_checks(node, input_sequence_and_position, input_sequence_and_position);

    const auto& inputs = node.get_ov_inputs();

    const auto& input_sequence = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr());
    OPENVINO_ASSERT(input_sequence, "SequenceAt: 'input' must be a sequence");

    auto position = inputs[1];
    OPENVINO_ASSERT(position.get_partial_shape().rank().compatible(0), "SequenceAt: 'position' input must be a scalar");

    const auto position_const = ov::util::get_constant_from_source(position);
    OPENVINO_ASSERT(position_const, "SequenceAt: 'position' input must be constant");

    const auto position_value = position_const->cast_vector<std::int64_t>()[0];

    const auto input_sequence_length = static_cast<std::int64_t>(input_sequence->get_sequence().size());

    const auto position_value_normalized = position_value < 0 ? position_value + input_sequence_length : position_value;
    OPENVINO_ASSERT(position_value_normalized >= 0 && position_value_normalized < input_sequence_length,
                    "SequenceAt: 'position' is out of bounds");

    return {input_sequence->get_sequence().at(position_value_normalized)};
}

/// @brief Registers the SequenceAt operator implementation in the ONNX frontend
/// @remark The operator is available since ONNX opset 11.
///         Registering as available since opset 1 for compatibility with existing tests.
ONNX_OP("SequenceAt", OPSET_SINCE(1), ai_onnx::opset_11::sequence_at);

}  // namespace opset_11

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
