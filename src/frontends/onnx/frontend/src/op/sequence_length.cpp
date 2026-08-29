// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_length.hpp"

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/type.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

/// @brief Implements the SequenceLength operator
/// @param node Input ONNX node with a single sequence input.
/// @return A scalar int64 tensor with the number of elements in the sequence.
ov::OutputVector sequence_length(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 1, 1);
    const auto& inputs = node.get_ov_inputs();

    // Fast path: input is a SequenceMark chain - resolve directly to a constant.
    if (const auto input_sequence = as_type_ptr<SequenceMark>(inputs[0].get_node_shared_ptr())) {
        const auto length = static_cast<std::int64_t>(input_sequence->get_sequence().size());
        return {ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {length})};
    }

    // Deferred path: emit a helper op that will be resolved by a later transformation.
    return {std::make_shared<ov::frontend::SequenceLength>(inputs[0])};
}

ONNX_OP("SequenceLength", OPSET_SINCE(1), ai_onnx::opset_11::sequence_length);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
