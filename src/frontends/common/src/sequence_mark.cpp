// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_mark.hpp"

namespace ov {
namespace frontend {

SequenceMark::SequenceMark(const ov::OutputVector& inputs) : ov::op::util::FrameworkNode(inputs, 1) {
    validate_and_infer_types();
}

void SequenceMark::validate_and_infer_types() {
    set_output_type(0, ov::element::dynamic, PartialShape::dynamic());
}

std::shared_ptr<Node> SequenceMark::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<SequenceMark>(inputs);
}

size_t SequenceMark::size() const {
    return get_input_size();
}

bool SequenceMark::empty() const {
    return get_input_size() == 0;
}

ov::Output<ov::Node> SequenceMark::get_element(size_t index) const {
    return input_value(index);
}

ov::OutputVector SequenceMark::get_sequence() const {
    return input_values();
}

}  // namespace frontend
}  // namespace ov
