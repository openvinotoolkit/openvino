// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_causal_conv1d_shape_inference.hpp"

namespace ov::op::internal {

PagedCausalConv1D::PagedCausalConv1D(const Output<Node>& input_embeds,
                                     const Output<Node>& conv_state_table,
                                     const Output<Node>& conv_weight,
                                     const Output<Node>& conv_bias,
                                     const Output<Node>& subsequence_begins,
                                     const Output<Node>& la_block_indices,
                                     const Output<Node>& la_block_indices_begins,
                                     const Output<Node>& processed_tokens,
                                     const Output<Node>& cache_interval)
    : Op({input_embeds,
          conv_state_table,
          conv_weight,
          conv_bias,
          subsequence_begins,
          la_block_indices,
          la_block_indices_begins,
          processed_tokens,
          cache_interval}) {
    constructor_validate_and_infer_types();
}

PagedCausalConv1D::PagedCausalConv1D(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedCausalConv1D::validate_and_infer_types() {
    OV_OP_SCOPE(PagedCausalConv1D_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 9);

    // input_embeds (0), conv_weight (2), conv_bias (3) participate in the convolution MAC and
    // therefore must share a common float element type; it also determines the output precision.
    // conv_state_table (1) is an in-place state cache and is allowed to use an independent float
    // element type so plugins can maintain a lower-precision state without breaking in-place semantics.
    ov::element::Type common_float_type = get_input_element_type(0);
    const bool float_types_merge =
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(2)) &&
        ov::element::Type::merge(common_float_type, common_float_type, get_input_element_type(3));
    NODE_VALIDATION_CHECK(this,
                          float_types_merge,
                          "PagedCausalConv1D expects input_embeds, conv_weight, and conv_bias to have the same "
                          "element type.");
    NODE_VALIDATION_CHECK(this,
                          common_float_type.is_dynamic() || common_float_type == ov::element::f32 ||
                              common_float_type == ov::element::f16 || common_float_type == ov::element::bf16,
                          "Float inputs must have f32, f16, or bf16 element type.");
    const auto& state_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          state_et.is_dynamic() || state_et == ov::element::f32 || state_et == ov::element::f16 ||
                              state_et == ov::element::bf16,
                          "Float inputs must have f32, f16, or bf16 element type.");
    for (size_t i = 4; i < 9; ++i) {
        const auto& et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              et.is_dynamic() || et == ov::element::i32 || et == ov::element::i64,
                              "Integer inputs must have i32 or i64 element type.");
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, common_float_type, output_shapes[0]);
}

bool PagedCausalConv1D::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedCausalConv1D_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> PagedCausalConv1D::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedCausalConv1D_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedCausalConv1D>(new_args);
}

}  // namespace ov::op::internal
