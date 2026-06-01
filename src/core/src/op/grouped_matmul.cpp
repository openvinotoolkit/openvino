// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include "grouped_matmul_shape_inference.hpp"
#include "itt.hpp"

namespace ov::op::v17 {

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b) : Op(OutputVector{mat_a, mat_b}) {
    constructor_validate_and_infer_types();
}

GroupedMatMul::GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b, const Output<Node>& offsets)
    : Op(OutputVector{mat_a, mat_b, offsets}) {
    constructor_validate_and_infer_types();
}

bool GroupedMatMul::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_GroupedMatMul_visit_attributes);
    return true;
}

void GroupedMatMul::validate_and_infer_types() {
    OV_OP_SCOPE(v17_GroupedMatMul_validate_and_infer_types);

    const auto& mat_a_et = get_input_element_type(0);
    const auto& mat_b_et = get_input_element_type(1);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, mat_a_et, mat_b_et),
                          "Arguments do not have the same element type (mat_a element type: ",
                          mat_a_et,
                          ", mat_b element type: ",
                          mat_b_et,
                          ").");

    if (get_input_size() == 3) {
        const auto& offsets_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              offsets_et.is_dynamic() || offsets_et == element::i32 || offsets_et == element::i64,
                              "Offsets element type must be i32 or i64. Got: ",
                              offsets_et);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> GroupedMatMul::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_GroupedMatMul_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    if (new_args.size() == 2) {
        return std::make_shared<GroupedMatMul>(new_args.at(0), new_args.at(1));
    } else {
        return std::make_shared<GroupedMatMul>(new_args.at(0), new_args.at(1), new_args.at(2));
    }
}

}  // namespace ov::op::v17
