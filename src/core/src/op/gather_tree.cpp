// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_tree.hpp"

#include "gather_tree_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
op::v1::GatherTree::GatherTree(const Output<Node>& step_ids,
                               const Output<Node>& parent_idx,
                               const Output<Node>& max_seq_len,
                               const Output<Node>& end_token)
    : Op({step_ids, parent_idx, max_seq_len, end_token}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v1::GatherTree::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_GatherTree_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::GatherTree>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::v1::GatherTree::validate_and_infer_types() {
    OV_OP_SCOPE(v1_GatherTree_validate_and_infer_types);

    const auto& step_ids_et = get_input_element_type(0);
    const auto& parent_idx_et = get_input_element_type(1);
    const auto& max_seq_len_et = get_input_element_type(2);
    const auto& end_token_et = get_input_element_type(3);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, step_ids_et, parent_idx_et) &&
                              element::Type::merge(result_et, result_et, max_seq_len_et) &&
                              element::Type::merge(result_et, result_et, end_token_et),
                          "Inputs must have the same element type. Got: step_ids (",
                          step_ids_et,
                          "), parent_idx_et (",
                          parent_idx_et,
                          "), max_seq_len (",
                          max_seq_len_et,
                          "), end_token (",
                          end_token_et,
                          ")");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, result_et, output_shapes[0]);
}
}  // namespace ov
