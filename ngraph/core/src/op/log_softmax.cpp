//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "ngraph/op/log_softmax.hpp"

#include "itt.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v5::LogSoftmax, "LogSoftmax", 5);

op::v5::LogSoftmax::LogSoftmax(const Output<Node>& arg, const int64_t axis)
    : Op({arg})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

bool op::v5::LogSoftmax::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v5_LogSoftmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v5::LogSoftmax::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v5_LogSoftmax_validate_and_infer_types);
    const PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < input_shape.rank().get_length() &&
                                  m_axis >= -input_shape.rank().get_length(),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

shared_ptr<Node> op::v5::LogSoftmax::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v5_LogSoftmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v5::LogSoftmax>(new_args.at(0), m_axis);
}
