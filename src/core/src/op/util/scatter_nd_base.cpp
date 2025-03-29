// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/scatter_nd_base.hpp"

#include "itt.hpp"
#include "scatter_nd_base_shape_inference.hpp"

constexpr int ov::op::util::ScatterNDBase::INPUTS;
constexpr int ov::op::util::ScatterNDBase::INDICES;
constexpr int ov::op::util::ScatterNDBase::UPDATES;

ov::op::util::ScatterNDBase::ScatterNDBase(const Output<Node>& data,
                                           const Output<Node>& indices,
                                           const Output<Node>& updates)
    : Op({data, indices, updates}) {
    constructor_validate_and_infer_types();
}

bool ov::op::util::ScatterNDBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_ScatterNDBase_visit_attributes);
    return true;
}

void ov::op::util::ScatterNDBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_ScatterNDBase_validate_and_infer_types);
    element::Type inputs_et = get_input_element_type(INPUTS);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    NODE_VALIDATION_CHECK(this,
                          indices_et.compatible(element::i32) || indices_et.compatible(element::i64),
                          "Indices element type must be i64 or i32");

    element::Type outputs_et = element::dynamic;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(outputs_et, inputs_et, updates_et),
                          "Updates element type must be the same as inputs");

    const auto& inputs = get_input_partial_shape(0);
    const auto& indices = get_input_partial_shape(1);
    const auto& updates = get_input_partial_shape(2);

    std::vector<ov::PartialShape> input_shapes = {inputs, indices, updates};
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, outputs_et, output_shapes[0]);
}
