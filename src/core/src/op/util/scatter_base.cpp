// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/scatter_base.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

ov::op::util::ScatterBase::ScatterBase(const Output<Node>& data,
                                       const Output<Node>& indices,
                                       const Output<Node>& updates,
                                       const Output<Node>& axis)
    : Op({data, indices, updates, axis}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::ScatterBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_ScatterBase_validate_and_infer_types);
    const auto& data_et = get_input_element_type(DATA);
    const auto& indices_et = get_input_element_type(INDICES);
    const auto& updates_et = get_input_element_type(UPDATES);
    const auto& axis_et = get_input_element_type(AXIS);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_integral_number(),
                          "Indices element type must be of an integral number type.");

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_et, updates_et),
                          "Element types for input data and updates do not match (data element type: ",
                          data_et,
                          ", updates element type: ",
                          updates_et,
                          ").");

    NODE_VALIDATION_CHECK(this, axis_et.is_integral_number(), "Axis element type must be of an integral number type.");

    const auto& data_shape = get_input_partial_shape(DATA);
    const auto& indices_shape = get_input_partial_shape(INDICES);
    const auto& updates_shape = get_input_partial_shape(UPDATES);
    const auto& axis_shape = get_input_partial_shape(AXIS);

    NODE_VALIDATION_CHECK(this,
                          axis_shape.compatible(PartialShape{}) || axis_shape.compatible(PartialShape{1}),
                          "Axis input shape is required to be scalar or 1D tensor. ",
                          "Got: ",
                          axis_shape);

    // Updates rank must be at indices rank + data rank - 1
    NODE_VALIDATION_CHECK(
        this,
        data_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() || updates_shape.rank().is_dynamic() ||
            updates_shape.rank().get_length() == indices_shape.rank().get_length() + data_shape.rank().get_length() - 1,
        "Updates rank is expected to be rank(indices) + rank(data) - 1.",
        " Got: rank(data) = ",
        data_shape.rank().get_length(),
        ", rank(indices) = ",
        indices_shape.rank().get_length(),
        ", rank(updates) = ",
        updates_shape.rank().get_length());

    if (data_shape.is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
    set_output_type(0, data_et, data_shape);

    if (data_shape.rank().is_dynamic())
        return;

    // Get axis value if possible.
    if (const auto& axis_const_input = ov::util::get_constant_from_source(input_value(AXIS))) {
        bool compatible = true;
        int64_t axis = axis_const_input->cast_vector<int64_t>().at(0);
        const int64_t data_rank = data_shape.rank().get_length();
        axis = ov::util::try_normalize_axis(axis, data_rank, *this);

        if (indices_shape.rank().is_static() && updates_shape.rank().is_static()) {
            int64_t indices_rank = indices_shape.rank().get_length();
            for (int64_t i = 0; i < indices_rank; ++i) {
                compatible = compatible && updates_shape[axis + i].compatible(indices_shape[i]);
            }

            // Check [d_0, d_1, ... d_(axis - 1)] updates dimensions
            for (int64_t i = 0; i < axis; ++i) {
                compatible = compatible && updates_shape[i].compatible(data_shape[i]);
            }
            // Check [d_(axis + k + 1), ..., d_n] updates dimensions
            for (int64_t i = axis + 1; i < data_rank; ++i) {
                compatible = compatible && updates_shape[indices_rank - 1 + i].compatible(data_shape[i]);
            }
        }
        NODE_VALIDATION_CHECK(this,
                              compatible,
                              "Updates shape must have appropriate dimensions equal to indices and "
                              "data dimensions. Updates shape:",
                              updates_shape,
                              ", data shape: ",
                              data_shape,
                              ", indices_shape: ",
                              indices_shape,
                              ", axis: ",
                              axis,
                              ".");
    }
}

bool ov::op::util::ScatterBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_ScatterBase_visit_attributes);
    return true;
}
