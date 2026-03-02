// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief ConcatFromSequence is a helper operation that represents concatenating
/// all tensors in a sequence along a specified axis.
/// This operation is used during frontend conversion to mark concat-from-sequence operations
/// that will be later transformed by SequenceConcatReplacer.
///
/// Inputs:
///   - input_sequence: The input sequence (SequenceMark, SequenceInsert, or Loop output)
///
/// Attributes:
///   - axis: The axis along which to concatenate (default: 0)
///   - new_axis: If true, creates a new axis for concatenation (like stack), otherwise
///               concatenates along existing axis (default: false)
///
/// Output:
///   - The concatenated tensor
class FRONTEND_API ConcatFromSequence : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("ConcatFromSequence", "util", ov::op::util::FrameworkNode);

    /// \brief Constructs ConcatFromSequence
    /// \param input_sequence The input sequence
    /// \param axis The axis along which to concatenate
    /// \param new_axis If true, creates a new axis for concatenation
    ConcatFromSequence(const Output<Node>& input_sequence, int64_t axis, bool new_axis = false);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    /// \brief Get the input sequence
    Output<Node> get_input_sequence() const {
        return input_value(0);
    }

    /// \brief Get the concatenation axis
    int64_t get_axis() const {
        return m_axis;
    }

    /// \brief Set the concatenation axis
    void set_axis(int64_t axis) {
        m_axis = axis;
    }

    /// \brief Check if new_axis mode is enabled
    bool get_new_axis() const {
        return m_new_axis;
    }

    /// \brief Set new_axis mode
    void set_new_axis(bool new_axis) {
        m_new_axis = new_axis;
    }

private:
    int64_t m_axis = 0;
    bool m_new_axis = false;
};

}  // namespace frontend
}  // namespace ov
