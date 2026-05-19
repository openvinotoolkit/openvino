// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_length.hpp"

namespace ov {
namespace frontend {

SequenceLength::SequenceLength(const Output<Node>& input_sequence) : FrameworkNode({input_sequence}, 1) {}

std::shared_ptr<Node> SequenceLength::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1, "SequenceLength requires 1 input");
    return std::make_shared<SequenceLength>(inputs[0]);
}

}  // namespace frontend
}  // namespace ov
