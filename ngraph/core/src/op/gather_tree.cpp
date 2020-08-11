//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/gather_tree.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::GatherTree::type_info;

op::v1::GatherTree::GatherTree(const Output<Node>& step_ids,
                               const Output<Node>& parent_idx,
                               const Output<Node>& max_seq_len,
                               const Output<Node>& end_token)
    : Op({step_ids, parent_idx, max_seq_len, end_token})
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::GatherTree::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::GatherTree>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool ngraph::op::v1::GatherTree::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v1::GatherTree::validate_and_infer_types()
{
    const auto& step_ids_rank = get_input_partial_shape(0);
    const auto& parent_idx_rank = get_input_partial_shape(1);
    const auto& max_seq_len_rank = get_input_partial_shape(2);
    const auto& end_token_rank = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          step_ids_rank.rank().is_dynamic() ||
                              step_ids_rank.rank().get_length() == 3,
                          "step_ids input rank must equal to 3 (step_ids rank: ",
                          step_ids_rank.rank().get_length(),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          parent_idx_rank.rank().is_dynamic() ||
                              parent_idx_rank.rank().get_length() == 3,
                          "parent_idx input rank must equal to 3 (parent_idx rank: ",
                          parent_idx_rank.rank().get_length(),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          max_seq_len_rank.rank().is_dynamic() ||
                              max_seq_len_rank.rank().get_length() == 1,
                          "max_seq_len input rank must equal to 1 (max_seq_len rank: ",
                          max_seq_len_rank.rank().get_length(),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          end_token_rank.rank().is_dynamic() ||
                              end_token_rank.rank().get_length() == 0,
                          "end_token input rank must be scalar (end_token rank: ",
                          end_token_rank.rank().get_length(),
                          ")");

    const auto& step_ids_et = get_input_element_type(0);
    set_output_type(0, step_ids_et, step_ids_rank);
}
