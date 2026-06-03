// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_erase.hpp"

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

/// @brief Implements the SequenceErase operator.
/// @param node Input ONNX node. Inputs: sequence and optional position.
///             When position is omitted, the last element is removed.
/// @return A new sequence with the requested element removed.
ov::OutputVector sequence_erase(const ov::frontend::onnx::Node& node) {
    constexpr auto min_inputs = 1;
    constexpr auto max_inputs = 2;
    common::default_op_checks(node, min_inputs, max_inputs);

    const auto& inputs = node.get_ov_inputs();

    if (inputs.size() == 2) {
        OPENVINO_ASSERT(inputs[1].get_partial_shape().rank().compatible(0),
                        "SequenceErase: 'position' input must be a scalar");
    }

    // Fast path: input is a SequenceMark with a constant (or omitted) position.
    if (const auto input_sequence = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr())) {
        auto seq = input_sequence->get_sequence();
        const auto length = static_cast<std::int64_t>(seq.size());
        std::int64_t position = length - 1;
        if (inputs.size() == 2) {
            const auto position_const = ov::util::get_constant_from_source(inputs[1]);
            if (!position_const) {
                // Position is dynamic - defer.
                return {std::make_shared<ov::frontend::SequenceErase>(inputs[0], inputs[1])};
            }
            position = position_const->cast_vector<std::int64_t>()[0];
        }
        const auto position_normalized = position < 0 ? position + length : position;
        OPENVINO_ASSERT(position_normalized >= 0 && position_normalized < length,
                        "SequenceErase: 'position' is out of bounds");
        seq.erase(seq.begin() + position_normalized);
        return {std::make_shared<ov::frontend::SequenceMark>(seq)};
    }

    // Deferred path: emit a helper op resolved by a later transformation.
    if (inputs.size() == 2) {
        return {std::make_shared<ov::frontend::SequenceErase>(inputs[0], inputs[1])};
    }
    return {std::make_shared<ov::frontend::SequenceErase>(inputs[0])};
}

ONNX_OP("SequenceErase", OPSET_SINCE(1), ai_onnx::opset_11::sequence_erase);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
