// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_insert.hpp"

namespace ov {
namespace frontend {

SequenceInsert::SequenceInsert(const Output<Node>& input_sequence, const Output<Node>& tensor)
    : FrameworkNode({input_sequence, tensor}, 1) {}

SequenceInsert::SequenceInsert(const Output<Node>& input_sequence,
                               const Output<Node>& tensor,
                               const Output<Node>& position)
    : FrameworkNode({input_sequence, tensor, position}, 1) {}

std::shared_ptr<Node> SequenceInsert::clone_with_new_inputs(const OutputVector& inputs) const {
    if (inputs.size() == 2) {
        return std::make_shared<SequenceInsert>(inputs[0], inputs[1]);
    } else if (inputs.size() == 3) {
        return std::make_shared<SequenceInsert>(inputs[0], inputs[1], inputs[2]);
    }
    OPENVINO_THROW("SequenceInsert requires 2 or 3 inputs");
}

}  // namespace frontend
}  // namespace ov
