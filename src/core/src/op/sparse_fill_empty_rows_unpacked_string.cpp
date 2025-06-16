// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "sparse_fill_empty_rows_unpacked_string_shape_inference.hpp"

namespace ov::op::v16 {

SparseFillEmptyRowsUnpackedString::SparseFillEmptyRowsUnpackedString(const Output<Node>& begins,
                                                                     const Output<Node>& ends,
                                                                     const Output<Node>& symbols,
                                                                     const Output<Node>& indices,
                                                                     const Output<Node>& dense_shape,
                                                                     const Output<Node>& default_value)
    : Op({begins, ends, symbols, indices, dense_shape, default_value}) {
    constructor_validate_and_infer_types();
}

void SparseFillEmptyRowsUnpackedString::validate_and_infer_types() {
    OV_OP_SCOPE(v16_SparseFillEmptyRowsUnpackedString_validate_and_infer_types);

    const auto& begins_element_type = get_input_element_type(0);
    const auto& ends_element_type = get_input_element_type(1);
    const auto& indices_element_type = get_input_element_type(3);
    const auto& dense_shape_element_type = get_input_element_type(4);
    element::Type merged_type;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, begins_element_type, ends_element_type),
                          "The element types of the begins and ends inputs must match. Got: ",
                          begins_element_type,
                          " and ",
                          ends_element_type);

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, merged_type, indices_element_type),
                          "The element type of indices must match begins and ends. Got: ",
                          merged_type,
                          " and ",
                          indices_element_type);

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, merged_type, dense_shape_element_type),
                          "The element type of dense_shape must match begins and ends. Got: ",
                          merged_type,
                          " and ",
                          dense_shape_element_type);

    NODE_VALIDATION_CHECK(this,
                          merged_type == element::i32 || merged_type == element::i64 || merged_type.is_dynamic(),
                          "The element type of the index inputs must be i32 or i64. Got: ",
                          merged_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, begins_element_type, output_shapes[0]);        // output_begins
    set_output_type(1, ends_element_type, output_shapes[1]);          // output_ends
    set_output_type(2, indices_element_type, output_shapes[2]);       // output_indices
    set_output_type(3, get_input_element_type(2), output_shapes[3]);  // output_symbols
    set_output_type(4, element::boolean, output_shapes[4]);           // empty_row_indicator
}

std::shared_ptr<Node> SparseFillEmptyRowsUnpackedString::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v16_SparseFillEmptyRowsUnpackedString_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SparseFillEmptyRowsUnpackedString>(new_args.at(0),
                                                               new_args.at(1),
                                                               new_args.at(2),
                                                               new_args.at(3),
                                                               new_args.at(4),
                                                               new_args.at(5));
}

}  // namespace ov::op::v16
