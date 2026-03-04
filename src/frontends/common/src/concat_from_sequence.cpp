// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/concat_from_sequence.hpp"

namespace ov {
namespace frontend {

ConcatFromSequence::ConcatFromSequence(const Output<Node>& input_sequence, int64_t axis, bool new_axis)
    : FrameworkNode({input_sequence}, 1),
      m_axis(axis),
      m_new_axis(new_axis) {}

std::shared_ptr<Node> ConcatFromSequence::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1, "ConcatFromSequence requires exactly 1 input");
    return std::make_shared<ConcatFromSequence>(inputs[0], m_axis, m_new_axis);
}

}  // namespace frontend
}  // namespace ov
