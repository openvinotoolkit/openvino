// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_selective_ssm.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_selective_ssm_shape_inference.hpp"

namespace ov::op::internal {

PagedSelectiveSSM::PagedSelectiveSSM(const Output<Node>& A,
                                     const Output<Node>& dt,
                                     const Output<Node>& B,
                                     const Output<Node>& x,
                                     const Output<Node>& C,
                                     const Output<Node>& recurrent_state_table,
                                     const Output<Node>& subsequence_begins,
                                     const Output<Node>& la_block_indices,
                                     const Output<Node>& la_block_indices_begins,
                                     const Output<Node>& num_processed_tokens,
                                     const Output<Node>& cache_interval)
    : Op({A,
          dt,
          B,
          x,
          C,
          recurrent_state_table,
          subsequence_begins,
          la_block_indices,
          la_block_indices_begins,
          num_processed_tokens,
          cache_interval}) {
    constructor_validate_and_infer_types();
}

PagedSelectiveSSM::PagedSelectiveSSM(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedSelectiveSSM::validate_and_infer_types() {
    OV_OP_SCOPE(PagedSelectiveSSM_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 11,
                          "PagedSelectiveSSM expects 11 inputs, but it has ",
                          get_input_size());

    ov::element::Type common_float_type = get_input_element_type(0);
    bool float_types_merge = true;
    for (size_t input = 1; input < 5; ++input) {
        float_types_merge &=
            ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(input));
    }
    NODE_VALIDATION_CHECK(this,
                          float_types_merge,
                          "PagedSelectiveSSM expects inputs A, dt, B, x, and C to have the same element type.");
    NODE_VALIDATION_CHECK(this,
                          common_float_type.is_dynamic() || common_float_type == ov::element::f32 ||
                              common_float_type == ov::element::f16 || common_float_type == ov::element::bf16,
                          "PagedSelectiveSSM data inputs must have f32, f16, or bf16 element type.");
    const auto& state_type = get_input_element_type(5);
    NODE_VALIDATION_CHECK(this,
                          state_type.is_dynamic() || state_type == ov::element::f32 || state_type == ov::element::f16 ||
                              state_type == ov::element::bf16,
                          "PagedSelectiveSSM recurrent_state_table must have f32, f16, or bf16 element type.");

    ov::element::Type common_index_type = get_input_element_type(6);
    bool index_types_merge = true;
    for (size_t input = 7; input < 11; ++input) {
        index_types_merge &=
            ov::element::Type::merge(common_index_type, common_index_type, get_input_element_type(input));
    }
    NODE_VALIDATION_CHECK(this,
                          index_types_merge,
                          "PagedSelectiveSSM expects all metadata inputs to have the same element type.");
    NODE_VALIDATION_CHECK(this,
                          common_index_type.is_dynamic() || common_index_type == ov::element::i32 ||
                              common_index_type == ov::element::i64,
                          "PagedSelectiveSSM metadata inputs must have i32 or i64 element type.");

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, common_float_type, output_shapes[0]);
}

bool PagedSelectiveSSM::visit_attributes(AttributeVisitor&) {
    OV_OP_SCOPE(PagedSelectiveSSM_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> PagedSelectiveSSM::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedSelectiveSSM_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedSelectiveSSM>(new_args);
}

}  // namespace ov::op::internal
