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
    // Infer element type: if all inputs have the same type, use that type
    element::Type output_type = element::dynamic;
    if (get_input_size() > 0) {
        output_type = get_input_element_type(0);
        for (size_t i = 1; i < get_input_size(); ++i) {
            if (get_input_element_type(i) != output_type) {
                output_type = element::dynamic;
                break;
            }
        }
    }

    // Infer shape: if all inputs are 0D (scalars), output shape is 1D with size = number of elements
    // If any input is more than 0D, we cannot determine the shape (depends on concat axis)
    PartialShape output_shape = PartialShape::dynamic();
    if (get_input_size() > 0) {
        bool all_scalars = true;
        for (size_t i = 0; i < get_input_size(); ++i) {
            const auto& input_shape = get_input_partial_shape(i);
            if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 0) {
                all_scalars = false;
                break;
            }
        }
        if (all_scalars) {
            output_shape = PartialShape{static_cast<int64_t>(get_input_size())};
        }
    }

    set_output_type(0, output_type, output_shape);
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
