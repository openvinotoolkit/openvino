// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceErase is a helper operation that represents removing one element
/// from a sequence. Emitted by frontends when the sequence input is not yet
/// resolvable to a SequenceMark.
///
/// Inputs:
///   - input_sequence: The input sequence.
///   - position (optional): The position of the element to remove. When omitted,
///                          the last element is removed (per the ONNX SequenceErase spec).
///
/// Output:
///   - The sequence with the element removed.
class FRONTEND_API SequenceErase : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceErase", "util", ov::op::util::FrameworkNode);

    explicit SequenceErase(const Output<Node>& input_sequence);
    SequenceErase(const Output<Node>& input_sequence, const Output<Node>& position);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    Output<Node> get_input_sequence() const {
        return input_value(0);
    }

    bool has_position() const {
        return get_input_size() > 1;
    }

    Output<Node> get_position() const {
        return has_position() ? input_value(1) : Output<Node>{};
    }
};

}  // namespace frontend
}  // namespace ov
