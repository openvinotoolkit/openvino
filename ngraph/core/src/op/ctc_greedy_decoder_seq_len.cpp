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

#include "ngraph/op/ctc_greedy_decoder_seq_len.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::CTCGreedyDecoderSeqLen, "CTCGreedyDecoderSeqLen", 6);

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len})
    , m_merge_repeated(merge_repeated)
    , m_classes_index_type(classes_index_type)
    , m_sequence_length_type(sequence_length_type)
{
    constructor_validate_and_infer_types();
}

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const Output<Node>& blank_index,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len, blank_index})
    , m_merge_repeated(merge_repeated)
    , m_classes_index_type(classes_index_type)
    , m_sequence_length_type(sequence_length_type)
{
    constructor_validate_and_infer_types();
}

void op::v6::CTCGreedyDecoderSeqLen::validate_and_infer_types()
{
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& seq_len_pshape = get_input_partial_shape(1);
    auto input_et = get_input_element_type(0);

    // output dynamic rank tensor if all inputs are of dynamic rank
    if (logits_pshape.rank().is_dynamic() && seq_len_pshape.rank().is_dynamic())
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
    if (seq_len_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              seq_len_pshape.rank().get_length() == 1,
                              "The rank of sequence len tensor must be equal to 1.");

    }

    // check optional input type: blank index
    if (get_input_size() == 3)
    {
        const auto& blank_index_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              blank_index_type.is_integral_number(),
                              "The blank index type is expected to be an integer type. Got: ",
                              blank_index_type);

        const auto& blank_index_shape = get_input_partial_shape(2);
        Shape blank_index_type_shape = blank_index_shape.to_shape();
        NODE_VALIDATION_CHECK(this,
                              blank_index_shape.is_dynamic() ||
                              ngraph::is_scalar(blank_index_type_shape) ||
                              (is_vector(blank_index_type_shape) && (blank_index_type_shape[0] == 1)),
                              "Expected 0D or 1D tensor for the 'blank_index' input. Got: ",
                              blank_index_type);
    }

    // validate input shapes and compute output shape
    ngraph::Dimension batch_size = Dimension::dynamic();
    ngraph::Dimension time_size = Dimension::dynamic();
    if (logits_pshape.rank().is_static())
    {
        if (logits_pshape[0].is_static())
        {
            batch_size = logits_pshape[0];
        }
        if (logits_pshape[1].is_static())
        {
            time_size = logits_pshape[1];
        }
    }
    if (seq_len_pshape.rank().is_static())
    {
        if (seq_len_pshape[0].is_static())
        {
            if (batch_size != Dimension::dynamic())
            {
                NODE_VALIDATION_CHECK(this,
                                      seq_len_pshape[0] == batch_size,
                                      "The first dimensions of input tensors must match.");
            }
            batch_size = seq_len_pshape[0];
        }
    }
    set_output_type(0, input_et, PartialShape{batch_size, time_size});
}

bool op::v6::CTCGreedyDecoderSeqLen::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("merge_repeated", m_merge_repeated);
    visitor.on_attribute("classes_index_type", m_classes_index_type);
    visitor.on_attribute("sequence_length_type", m_sequence_length_type);
    return true;
}

shared_ptr<Node>
    op::v6::CTCGreedyDecoderSeqLen::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);

    if (new_args.size() == 2)
    {
        return make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                   new_args.at(1),
                                                   m_merge_repeated,
                                                   m_classes_index_type,
                                                   m_sequence_length_type);
    }
    else if (new_args.size() == 3)
    {
        return make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(2),
                                                   m_merge_repeated,
                                                   m_classes_index_type,
                                                   m_sequence_length_type);
    }
    else
    {
        throw ngraph_error("Incorrect number of arguments");
    }
}
