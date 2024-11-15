// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/search_sorted.hpp>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "search_sorted_shape_inference.hpp"

namespace ov {
namespace op {
namespace v15 {

SearchSorted::SearchSorted(const Output<Node>& sorted_sequence,
                           const Output<Node>& values,
                           bool right_mode,
                           const element::Type& output_type)
    : Op({sorted_sequence, values}),
      m_right_mode(right_mode),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

void SearchSorted::validate_and_infer_types() {
    OV_OP_SCOPE(v15_SearchSorted_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(get_input_element_type(1)),
                          "Sorted sequence and values must have the same element type.");
    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i32 || m_output_type == element::i64,
                          "The element type of the last output can only be set to i32 or i64.");

    const auto& output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, m_output_type, output_shapes[0]);
}

bool SearchSorted::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_SearchSorted_visit_attributes);
    visitor.on_attribute("right_mode", m_right_mode);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<Node> SearchSorted::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_SearchSorted_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SearchSorted>(new_args.at(0), new_args.at(1), get_right_mode(), get_output_type_attr());
}
}  // namespace v15
}  // namespace op
}  // namespace ov