// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "sparse_fill_empty_rows_shape_inference.hpp"

namespace ov::op::v16 {

SparseFillEmptyRows::SparseFillEmptyRows(const Output<Node>& values,
                                         const Output<Node>& dense_shape,
                                         const Output<Node>& indices,
                                         const Output<Node>& default_value)
    : Op({values, dense_shape, indices, default_value}) {
    constructor_validate_and_infer_types();
}

void SparseFillEmptyRows::validate_and_infer_types() {
    OV_OP_SCOPE(v16_SparseFillEmptyRows_validate_and_infer_types);

    const auto& indices_element_type = get_input_element_type(2);
    const auto& dense_shape_element_type = get_input_element_type(1);
    element::Type merged_type;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, indices_element_type, dense_shape_element_type),
                          "The element types of the dense_shape and indices inputs must match. Got: ",
                          dense_shape_element_type,
                          " and ",
                          indices_element_type);

    NODE_VALIDATION_CHECK(this,
                          merged_type == element::i32 || merged_type == element::i64 || merged_type.is_dynamic(),
                          "The element type of the indices and dense_shape inputs must be i32 or i64. Got: ",
                          merged_type);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, indices_element_type, output_shapes[0]);
    set_output_type(1, get_input_element_type(0), output_shapes[1]);
    set_output_type(2, element::boolean, output_shapes[2]);
}

std::shared_ptr<Node> SparseFillEmptyRows::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v16_SparseFillEmptyRows_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SparseFillEmptyRows>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

}  // namespace ov::op::v16
