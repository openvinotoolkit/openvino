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

#include "ngraph/op/read_value.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReadValue::type_info;

op::ReadValue::ReadValue(const Output<Node>& new_value, const std::string& variable_id)
    : Op({new_value})
    , m_variable_id(variable_id)
{
    constructor_validate_and_infer_types();
}

void op::ReadValue::validate_and_infer_types()
{
    auto arg_t = get_input_element_type(0);
    auto output_shape = get_input_partial_shape(0);

    VariableInfo info = {output_shape, arg_t, m_variable_id};
    m_variable = std::make_shared<Variable>(info);
    set_output_type(0, arg_t, output_shape);
}

shared_ptr<Node> op::ReadValue::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReadValue>(new_args.at(0), m_variable_id);
}

bool op::v3::ReadValue::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}
