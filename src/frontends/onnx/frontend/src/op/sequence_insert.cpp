// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_insert.hpp"

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

ov::OutputVector sequence_insert(const ov::frontend::onnx::Node& node) {
    constexpr auto min_inputs = 2;
    constexpr auto max_inputs = 3;
    common::default_op_checks(node, min_inputs, max_inputs);

    const auto& inputs = node.get_ov_inputs();
    const auto& sequence_input = inputs[0];
    const auto& to_insert = inputs[1];

    // Create a SequenceInsert helper op that will be resolved by a transformation pass.
    // Wrap the result in a SequenceMark so that downstream consumers (e.g. ConcatFromSequence
    // or SequenceAt) can use SequenceMark::get_sequence() to walk through a chain of
    // SequenceInsert nodes and collect all inserted elements. This mirrors what the PyTorch
    // frontend does for aten::append. Without the wrapping SequenceMark, a chain of
    // SequenceInsert nodes feeding into ConcatFromSequence (as produced for models like
    // ConvNeXt/Swin) cannot be matched by the SequenceConcatReplacer pattern.
    std::shared_ptr<ov::frontend::SequenceInsert> seq_insert;
    if (inputs.size() == 3) {
        seq_insert = std::make_shared<ov::frontend::SequenceInsert>(sequence_input, to_insert, inputs[2]);
    } else {
        seq_insert = std::make_shared<ov::frontend::SequenceInsert>(sequence_input, to_insert);
    }
    auto sequence = std::make_shared<ov::frontend::SequenceMark>(ov::OutputVector{seq_insert});
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("SequenceInsert");
    sequence->set_attrs(attrs);
    return {sequence};
}

ONNX_OP("SequenceInsert", OPSET_SINCE(1), ai_onnx::opset_11::sequence_insert);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
