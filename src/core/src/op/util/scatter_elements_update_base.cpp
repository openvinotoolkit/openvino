// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/scatter_elements_update_base.hpp"

#include <scatter_elements_update_shape_inference.hpp>

#include "itt.hpp"

ov::op::util::ScatterElementsUpdateBase::ScatterElementsUpdateBase(const Output<Node>& data,
                                                                   const Output<Node>& indices,
                                                                   const Output<Node>& updates,
                                                                   const Output<Node>& axis)
    : Op({data, indices, updates, axis}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::ScatterElementsUpdateBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_ScatterElementsUpdateBase_validate_and_infer_types);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const element::Type& data_et = get_input_element_type(0);
    const element::Type& indices_et = get_input_element_type(1);
    const element::Type& updates_et = get_input_element_type(2);
    const element::Type& axis_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_integral(),
                          "Indices element type must be integral_number, but is: ",
                          indices_et);

    NODE_VALIDATION_CHECK(this, axis_et.is_integral(), "Axis element type must be integral_number, but is: ", axis_et);

    element::Type merged_type;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, data_et, updates_et),
                          "Data type and updates type are required to be the same. ",
                          "Got: ",
                          data_et,
                          " and: ",
                          updates_et);
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_input_element_type(0), output_shape);
    if (output_shape.is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
}
