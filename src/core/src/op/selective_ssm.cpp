// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/selective_ssm.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "selective_ssm_shape_inference.hpp"

namespace ov::op::internal {

SelectiveSSM::SelectiveSSM(const Output<Node>& A,
                           const Output<Node>& dt,
                           const Output<Node>& B,
                           const Output<Node>& x,
                           const Output<Node>& C,
                           const Output<Node>& recurrent_state)
    : Op({A, dt, B, x, C, recurrent_state}) {
    constructor_validate_and_infer_types();
}

SelectiveSSM::SelectiveSSM(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void SelectiveSSM::validate_and_infer_types() {
    OV_OP_SCOPE(SelectiveSSM_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 6, "SelectiveSSM expects 6 inputs, but it has ", get_input_size());

    ov::element::Type common_float_type = get_input_element_type(0);
    bool float_types_merge = true;
    for (size_t input = 1; input < 6; ++input) {
        float_types_merge &=
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(input));
    }
    NODE_VALIDATION_CHECK(this, float_types_merge, "SelectiveSSM expects all inputs to have the same element type.");
    NODE_VALIDATION_CHECK(this,
                          common_float_type.is_dynamic() || common_float_type == ov::element::f32 ||
                              common_float_type == ov::element::f16 || common_float_type == ov::element::bf16,
                          "SelectiveSSM inputs must have f32, f16, or bf16 element type.");

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, common_float_type, output_shapes[0]);
    set_output_type(1, common_float_type, output_shapes[1]);
}

bool SelectiveSSM::visit_attributes(AttributeVisitor&) {
    OV_OP_SCOPE(SelectiveSSM_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> SelectiveSSM::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SelectiveSSM>(new_args);
}

}  // namespace ov::op::internal
