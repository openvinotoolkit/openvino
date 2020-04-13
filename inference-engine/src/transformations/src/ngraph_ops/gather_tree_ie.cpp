// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/gather_tree_ie.hpp"

#include <memory>
#include <string>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GatherTreeIE::type_info;

op::GatherTreeIE::GatherTreeIE(const Output<Node>& step_ids,
                               const Output<Node>& parent_idx,
                               const Output<Node>& max_seq_len,
                               const Output<Node>& end_token)
        : Op({step_ids, parent_idx, max_seq_len, end_token}) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::GatherTreeIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<GatherTreeIE>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::GatherTreeIE::validate_and_infer_types() {
    const auto& step_ids_rank = get_input_partial_shape(0);
    const auto& parent_idx_rank = get_input_partial_shape(1);
    const auto& max_seq_len_rank = get_input_partial_shape(2);
    const auto& end_token_rank = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          step_ids_rank.rank().is_dynamic() ||
                          static_cast<size_t>(step_ids_rank.rank()) == 3,
                          "step_ids input rank must equal to 3 (step_ids rank: ",
                          static_cast<size_t>(step_ids_rank.rank()),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          parent_idx_rank.rank().is_dynamic() ||
                          static_cast<size_t>(parent_idx_rank.rank()) == 3,
                          "parent_idx input rank must equal to 3 (parent_idx rank: ",
                          static_cast<size_t>(parent_idx_rank.rank()),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          max_seq_len_rank.rank().is_dynamic() ||
                          static_cast<size_t>(max_seq_len_rank.rank()) == 1,
                          "max_seq_len input rank must equal to 1 (max_seq_len rank: ",
                          static_cast<size_t>(max_seq_len_rank.rank()),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          end_token_rank.rank().is_dynamic() ||
                          static_cast<size_t>(end_token_rank.rank()) == 1,
                          "end_token input rank must be scalar (end_token rank: ",
                          static_cast<size_t>(end_token_rank.rank()),
                          ")");

    const auto& step_ids_et = get_input_element_type(0);
    set_output_type(0, step_ids_et, step_ids_rank);
}
