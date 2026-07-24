// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceAt is a helper operation that represents extracting a single element
/// from a sequence at a given position. It is emitted by frontends when the sequence
/// input is not yet resolvable to a SequenceMark (e.g., comes from an If/Loop output)
/// and is later resolved by SequenceIfReplacer or similar passes.
///
/// Inputs:
///   - input_sequence: The input sequence (typically the output of an If/Loop whose
///                     body produces a SequenceMark/SequenceInsert chain).
///   - position: The position (scalar tensor) of the element to extract.
///
/// Output:
///   - The tensor at the requested position.
class FRONTEND_API SequenceAt : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceAt", "util", ov::op::util::FrameworkNode);

    SequenceAt(const Output<Node>& input_sequence, const Output<Node>& position);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    Output<Node> get_input_sequence() const {
        return input_value(0);
    }

    Output<Node> get_position() const {
        return input_value(1);
    }
};

}  // namespace frontend
}  // namespace ov
