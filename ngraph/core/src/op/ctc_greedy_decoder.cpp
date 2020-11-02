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

#include "ngraph/op/ctc_greedy_decoder.hpp"

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
    auto input_et = get_input_element_type(0);
    if (get_input_partial_shape(0).is_static())
    {
        Shape input_shape = get_input_partial_shape(0).to_shape();
        NODE_VALIDATION_CHECK(this,
                              input_shape.size() >= 3,
                              "CTCGreedyDecoder expects 3 or more dimensions for input. Got ",
                              input_shape.size());
        // TODO: Add more validation checks for seq_len

        set_output_type(0, input_et, Shape{input_shape[1], input_shape[0], 1, 1});
    }
    else
    {
        set_output_type(0, input_et, PartialShape::dynamic());
    }
}

bool op::CTCGreedyDecoder::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("ctc_merge_repeated", m_ctc_merge_repeated);
    return true;
}

shared_ptr<Node> op::CTCGreedyDecoder::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CTCGreedyDecoder>(new_args.at(0), new_args.at(1), m_ctc_merge_repeated);
}
