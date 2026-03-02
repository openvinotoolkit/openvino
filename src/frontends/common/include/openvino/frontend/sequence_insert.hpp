// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceInsert is a helper operation that represents inserting a tensor into a sequence.
/// This operation is used during frontend conversion to mark sequence insert operations
/// that will be later transformed by SequenceConcatReplacer.
///
/// Inputs:
///   - input_sequence: The input sequence (can be SequenceMark or another SequenceInsert)
///   - tensor: The tensor to insert into the sequence
///   - position (optional): The position at which to insert the tensor
///
/// Output:
///   - The updated sequence with the tensor inserted
class FRONTEND_API SequenceInsert : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceInsert", "util", ov::op::util::FrameworkNode);

    /// \brief Constructs SequenceInsert with sequence and tensor inputs
    /// \param input_sequence The input sequence
    /// \param tensor The tensor to insert
    SequenceInsert(const Output<Node>& input_sequence, const Output<Node>& tensor);

    /// \brief Constructs SequenceInsert with sequence, tensor, and position inputs
    /// \param input_sequence The input sequence
    /// \param tensor The tensor to insert
    /// \param position The position at which to insert
    SequenceInsert(const Output<Node>& input_sequence, const Output<Node>& tensor, const Output<Node>& position);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    /// \brief Get the input sequence
    Output<Node> get_input_sequence() const {
        return input_value(0);
    }

    /// \brief Get the tensor being inserted
    Output<Node> get_tensor() const {
        return input_value(1);
    }

    /// \brief Check if position input is provided
    bool has_position() const {
        return get_input_size() > 2;
    }

    /// \brief Get the position input (if provided)
    Output<Node> get_position() const {
        return has_position() ? input_value(2) : Output<Node>{};
    }
};

}  // namespace frontend
}  // namespace ov
