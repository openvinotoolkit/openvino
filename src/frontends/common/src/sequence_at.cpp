// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_at.hpp"

namespace ov {
namespace frontend {

SequenceAt::SequenceAt(const Output<Node>& input_sequence, const Output<Node>& position)
    : FrameworkNode({input_sequence, position}, 1) {}

std::shared_ptr<Node> SequenceAt::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 2, "SequenceAt requires 2 inputs");
    return std::make_shared<SequenceAt>(inputs[0], inputs[1]);
}

}  // namespace frontend
}  // namespace ov
