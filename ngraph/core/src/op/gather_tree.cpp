// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather_tree.hpp"

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_RTTI_DEFINITION(op::v1::GatherTree, "GatherTree", 1);

op::v1::GatherTree::GatherTree(const Output<Node>& step_ids,
                               const Output<Node>& parent_idx,
                               const Output<Node>& max_seq_len,
                               const Output<Node>& end_token)
    : Op({step_ids, parent_idx, max_seq_len, end_token}) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::GatherTree::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_GatherTree_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::GatherTree>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::GatherTree::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_GatherTree_visit_attributes);
    return true;
}

void op::v1::GatherTree::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_GatherTree_validate_and_infer_types);

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

    const auto& step_ids_pshape = get_input_partial_shape(0);
    const auto& parent_idx_pshape = get_input_partial_shape(1);
    const auto& max_seq_len_pshape = get_input_partial_shape(2);
    const auto& end_token_pshape = get_input_partial_shape(3);

    PartialShape result_pshape{PartialShape::dynamic()};
    NODE_VALIDATION_CHECK(this,
                          PartialShape::merge_into(result_pshape, step_ids_pshape) &&
                              PartialShape::merge_into(result_pshape, parent_idx_pshape) &&
                              result_pshape.rank().compatible(3),
                          "step_ids and parent_idx inputs must have the same shape with rank 3. Got: ",
                          step_ids_pshape,
                          " and ",
                          parent_idx_pshape,
                          ", respectively");

    NODE_VALIDATION_CHECK(this,
                          max_seq_len_pshape.rank().compatible(1),
                          "max_seq_len input must have rank 1. Got: ",
                          max_seq_len_pshape);

    if (result_pshape.rank().is_static() && max_seq_len_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              Dimension::merge(result_pshape[1], result_pshape[1], max_seq_len_pshape[0]),
                              "Number of elements of max_seq_len input must match BATCH_SIZE dimension of "
                              "step_ids/parent_idx inputs. Got: ",
                              result_pshape[1],
                              " and ",
                              max_seq_len_pshape[0],
                              ", respectively");
    }

    NODE_VALIDATION_CHECK(this,
                          end_token_pshape.rank().compatible(0),
                          "end_token input must be scalar. Got: ",
                          end_token_pshape);

    set_output_type(0, result_et, result_pshape);
}
