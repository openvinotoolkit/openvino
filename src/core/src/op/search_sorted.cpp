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

SearchSorted::SearchSorted(const Output<Node>& sorted_sequence, const Output<Node>& values, bool right_mode)
    : Op({sorted_sequence, values}),
      m_right_mode(right_mode) {
    constructor_validate_and_infer_types();
}

bool SearchSorted::validate() const {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(get_input_element_type(1)),
                          "Sorted sequence and values must have the same element type.");

    const auto& sorted_shape = get_input_partial_shape(0);
    const auto& values_shape = get_input_partial_shape(1);

    if (sorted_shape.rank().is_static() && values_shape.rank().is_static() && sorted_shape.rank().get_length() > 1) {
        NODE_VALIDATION_CHECK(this,
                              sorted_shape.rank().get_length() == values_shape.rank().get_length(),
                              "Sorted sequence and values have different ranks.");

        for (int64_t i = 0; i < sorted_shape.rank().get_length() - 1; ++i) {
            NODE_VALIDATION_CHECK(this,
                                  sorted_shape[i].compatible(values_shape[i]),
                                  "Sorted sequence and values has different ",
                                  i,
                                  " dimension.");
        }
    }

    return true;
}

void SearchSorted::validate_and_infer_types() {
    OV_OP_SCOPE(v15_SearchSorted_validate_and_infer_types);
    const auto& output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, ov::element::i64, output_shapes[0]);
}

bool SearchSorted::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_SearchSorted_visit_attributes);
    visitor.on_attribute("right_mode", m_right_mode);
    return true;
}

std::shared_ptr<Node> SearchSorted::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_SearchSorted_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SearchSorted>(new_args.at(0), new_args.at(1), get_right_mode());
}
}  // namespace v15
}  // namespace op
}  // namespace ov