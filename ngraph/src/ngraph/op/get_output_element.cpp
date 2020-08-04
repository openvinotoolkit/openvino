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

#include <sstream>

#include "ngraph/op/get_output_element.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GetOutputElement::type_info;

op::GetOutputElement::GetOutputElement(const shared_ptr<Node>& arg, size_t n)
    : Op({arg->output(n)})
    , m_n{n}
{
    constructor_validate_and_infer_types();
}

void op::GetOutputElement::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          m_n < input_value(0).get_node()->get_output_size(),
                          "Output at index ",
                          m_n,
                          " requested, but node has only ",
                          get_input_size(),
                          " inputs.");

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::GetOutputElement::clone_with_new_inputs(const OutputVector& inputs) const
{
    auto& value = inputs.at(0);
    return make_shared<op::GetOutputElement>(value.get_node_shared_ptr(), value.get_index());
}

Output<Node> op::GetOutputElement::get_as_output() const
{
    return input_value(0);
}

NodeVector op::get_output_elements(const shared_ptr<Node>& mon)
{
    NodeVector goes(mon->get_output_size());
    for (auto o : mon->outputs())
    {
        NGRAPH_SUPPRESS_DEPRECATED_START
        goes.at(o.get_index()) = o.as_single_output_node();
        NGRAPH_SUPPRESS_DEPRECATED_END
    }
    return goes;
}
