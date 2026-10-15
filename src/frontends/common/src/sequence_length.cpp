// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_length.hpp"

#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace frontend {

SequenceLength::SequenceLength(const Output<Node>& input_sequence) : FrameworkNode({input_sequence}, 1) {
    validate_and_infer_types();
}

void SequenceLength::validate_and_infer_types() {
    set_output_type(0, ov::element::i64, ov::Shape{});
}

std::shared_ptr<Node> SequenceLength::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.size() == 1, "SequenceLength requires 1 input");
    return std::make_shared<SequenceLength>(inputs[0]);
}

}  // namespace frontend
}  // namespace ov
