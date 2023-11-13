// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/scatter_elements_update_base.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "scatter_elements_update_shape_inference.hpp"

namespace ov {
namespace op {

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
    element::Type out_et = get_input_element_type(0);
    std::ignore = element::Type::merge(out_et, get_input_element_type(0), get_input_element_type(2));
    set_output_type(0, out_et, output_shape);
    if (output_shape.is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
}

bool op::util::ScatterElementsUpdateBase::has_evaluate() const {
    OV_OP_SCOPE(util_ScatterElementsUpdateBase_has_evaluate);

    switch (get_output_element_type(0)) {
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        break;
    default:
        return false;
    }

    return is_supported_index_input_element_type();
}

bool op::util::ScatterElementsUpdateBase::is_supported_index_input_element_type() const {
    switch (get_input_element_type(1)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}

bool op::util::ScatterElementsUpdateBase::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && ov::default_lower_bound_evaluator(this, output_values);
}

bool op::util::ScatterElementsUpdateBase::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && ov::default_upper_bound_evaluator(this, output_values);
}

bool op::util::ScatterElementsUpdateBase::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_label);

    OPENVINO_SUPPRESS_DEPRECATED_START
    return ov::default_label_evaluator(this, {0, 2}, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

OPENVINO_SUPPRESS_DEPRECATED_START
int64_t op::util::ScatterElementsUpdateBase::get_normalized_axis(const HostTensorVector& inputs) const {
    OPENVINO_ASSERT(inputs[3]->get_element_type().is_integral_number(), "axis element type is not integral data type");

    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = host_tensor_2_vector<int64_t>(inputs[3])[0];
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto& input_rank = get_input_partial_shape(0).rank();
    int64_t normalized_axis = axis;

    if (normalized_axis < 0) {
        if (input_rank.is_static()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            normalized_axis = ov::normalize_axis(this, axis, input_rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else {
            OPENVINO_SUPPRESS_DEPRECATED_START
            normalized_axis = ov::normalize_axis(this, axis, static_cast<int64_t>(inputs[0]->get_shape().size()));
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
    }
    return normalized_axis;
}
}  // namespace op
}  // namespace ov
