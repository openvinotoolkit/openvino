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

#include "ngraph/op/assign.hpp"
#include <ops.hpp>
#include "ngraph/op/read_value.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::Assign::type_info;

op::v3::Assign::Assign(const Output<Node>& new_value, const std::string& variable_id)
    : Op({new_value})
    , m_variable_id(variable_id)
{
    constructor_validate_and_infer_types();
}

void op::v3::Assign::validate_and_infer_types()
{
    auto value = input_value(0);
    auto arg_t = get_input_element_type(0);
    auto output_shape = get_input_partial_shape(0);
    if (!m_variable)
    {
        NodeVector start_nodes;
        for (const auto& input : inputs())
        {
            start_nodes.push_back(input.get_source_output().get_node_shared_ptr());
        }
        auto nodes = topological_sort(start_nodes);
        for (const auto& node : nodes)
        {
            if (auto read_value = as_type_ptr<op::v3::ReadValue>(node))
            {
                if (read_value->get_variable_id() == m_variable_id)
                    m_variable = read_value->get_variable();
            }
        }
        NODE_VALIDATION_CHECK(
            this, m_variable != nullptr, "Can't find variable with id = ", m_variable_id);
    }

    auto variable_info = m_variable->get_info();
    NODE_VALIDATION_CHECK(this,
                          m_variable_id == variable_info.variable_id,
                          "Variables identifiers are inconsistent.");
    NODE_VALIDATION_CHECK(
        this, arg_t == variable_info.data_type, "Variables types are inconsistent.");
    NODE_VALIDATION_CHECK(this,
                          output_shape == variable_info.data_shape,
                          "Variables output shapes are inconsistent.");

    set_output_type(0, arg_t, output_shape);
}

shared_ptr<Node> op::v3::Assign::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Assign>(new_args.at(0), m_variable_id);
}

bool op::v3::Assign::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("variable_id", m_variable_id);
    return true;
}
