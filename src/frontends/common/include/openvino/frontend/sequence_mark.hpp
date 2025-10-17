// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceMark serves to mark places that require a sequence type propagation
class FRONTEND_API SequenceMark : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceMark", "util", ov::op::util::FrameworkNode);

    /// \brief Constructors for SequenceMark node forwarded from FrameworkNode
    using ov::op::util::FrameworkNode::FrameworkNode;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    /// @brief Get the sequence of tensors
    /// @return Collection of inputs representing the sequence
    ov::OutputVector get_sequence() const;
};

inline ov::OutputVector SequenceMark::get_sequence() const {
    return input_values();
}

}  // namespace frontend
}  // namespace ov
