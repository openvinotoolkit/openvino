// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_dynamic_unit_split.hpp"

namespace ov {
namespace frontend {

SequenceDynamicUnitSplit::SequenceDynamicUnitSplit(const Output<Node>& data, int64_t axis)
    : FrameworkNode({data}, 1),
      m_axis(axis) {}

std::shared_ptr<Node> SequenceDynamicUnitSplit::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1, "SequenceDynamicUnitSplit requires 1 input");
    return std::make_shared<SequenceDynamicUnitSplit>(inputs[0], m_axis);
}

}  // namespace frontend
}  // namespace ov
