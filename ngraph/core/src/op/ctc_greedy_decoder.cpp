//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/ctc_greedy_decoder.hpp"
#include "itt.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CTCGreedyDecoder::type_info;

op::CTCGreedyDecoder::CTCGreedyDecoder(const Output<Node>& input,
                                       const Output<Node>& seq_len,
                                       const bool ctc_merge_repeated)
    : Op({input, seq_len})
    , m_ctc_merge_repeated(ctc_merge_repeated)
{
    constructor_validate_and_infer_types();
}

void op::CTCGreedyDecoder::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_CTCGreedyDecoder_validate_and_infer_types);
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& seq_mask_pshape = get_input_partial_shape(1);
    auto input_et = get_input_element_type(0);

    // output dynamic rank tensor if all inputs are of dynamic rank
    if (logits_pshape.rank().is_dynamic() && seq_mask_pshape.rank().is_dynamic())
    {
        set_output_type(
            0, input_et, PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1});
    }

    // check ranks of input tensors
    if (logits_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              logits_pshape.rank().get_length() == 3,
                              "The rank of logits tensor must be equal to 3.");
    }
    if (seq_mask_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              seq_mask_pshape.rank().get_length() == 2,
                              "The rank of sequence mask tensor must be equal to 2.");
    }

    // validate input shapes and compute output shape
    ngraph::Dimension batch_size = Dimension::dynamic();
    ngraph::Dimension time_size = Dimension::dynamic();
    if (logits_pshape.rank().is_static())
    {
        if (logits_pshape[0].is_static())
        {
            time_size = logits_pshape[0];
        }
        if (logits_pshape[1].is_static())
        {
            batch_size = logits_pshape[1];
        }
    }
    if (seq_mask_pshape.rank().is_static())
    {
        if (seq_mask_pshape[0].is_static())
        {
            if (time_size != Dimension::dynamic())
            {
                NODE_VALIDATION_CHECK(this,
                                      seq_mask_pshape[0] == time_size,
                                      "The first dimensions of input tensors must match.");
            }
            time_size = seq_mask_pshape[0];
        }
        if (seq_mask_pshape[1].is_static())
        {
            if (batch_size != Dimension::dynamic())
            {
                NODE_VALIDATION_CHECK(this,
                                      seq_mask_pshape[1] == batch_size,
                                      "The second dimensions of input tensors must match.");
            }
            batch_size = seq_mask_pshape[1];
        }
    }
    set_output_type(0, input_et, PartialShape{batch_size, time_size, 1, 1});
}

bool op::CTCGreedyDecoder::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_CTCGreedyDecoder_visit_attributes);
    visitor.on_attribute("ctc_merge_repeated", m_ctc_merge_repeated);
    return true;
}

shared_ptr<Node> op::CTCGreedyDecoder::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_CTCGreedyDecoder_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<CTCGreedyDecoder>(new_args.at(0), new_args.at(1), m_ctc_merge_repeated);
}
