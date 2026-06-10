// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceLength is a helper operation that represents the length of a sequence.
/// Emitted by frontends when the sequence input is not yet resolvable to a SequenceMark.
///
/// Inputs:
///   - input_sequence: The input sequence.
///
/// Output:
///   - Scalar int64 tensor with the number of elements in the sequence.
class FRONTEND_API SequenceLength : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceLength", "util", ov::op::util::FrameworkNode);

    explicit SequenceLength(const Output<Node>& input_sequence);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    Output<Node> get_input_sequence() const {
        return input_value(0);
    }
};

}  // namespace frontend
}  // namespace ov
