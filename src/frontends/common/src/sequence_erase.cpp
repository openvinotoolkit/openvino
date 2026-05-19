// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_erase.hpp"

namespace ov {
namespace frontend {

SequenceErase::SequenceErase(const Output<Node>& input_sequence) : FrameworkNode({input_sequence}, 1) {}

SequenceErase::SequenceErase(const Output<Node>& input_sequence, const Output<Node>& position)
    : FrameworkNode({input_sequence, position}, 1) {}

std::shared_ptr<Node> SequenceErase::clone_with_new_inputs(const OutputVector& inputs) const {
    if (inputs.size() == 1) {
        return std::make_shared<SequenceErase>(inputs[0]);
    } else if (inputs.size() == 2) {
        return std::make_shared<SequenceErase>(inputs[0], inputs[1]);
    }
    OPENVINO_THROW("SequenceErase requires 1 or 2 inputs");
}

}  // namespace frontend
}  // namespace ov
