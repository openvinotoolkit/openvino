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

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/parameter.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Parameter::type_info;

op::Parameter::Parameter(const element::Type& element_type,
                         const PartialShape& pshape,
                         const bool cacheable)
    : m_cacheable(cacheable)
    , m_partial_shape(pshape)
    , m_element_type(element_type)
    , m_is_relevant_to_shapes(false)
{
    constructor_validate_and_infer_types();
}

bool op::Parameter::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("cacheable", m_cacheable);
    visitor.on_attribute("shape", m_partial_shape);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

void op::Parameter::validate_and_infer_types()
{
    Op::validate_and_infer_types();
    set_output_type(0, m_element_type, m_partial_shape);
}

shared_ptr<Node> op::Parameter::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Parameter>(m_element_type, m_partial_shape);
}

void op::Parameter::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                      const OutputVector& deltas)
{
    auto delta = deltas.at(0);
}

bool op::Parameter::is_relevant_to_shapes() const
{
    return m_is_relevant_to_shapes;
}

void op::Parameter::set_is_relevant_to_shapes(bool is_relevant)
{
    m_is_relevant_to_shapes = is_relevant;
}

constexpr DiscreteTypeInfo AttributeAdapter<ParameterVector>::type_info;

AttributeAdapter<ParameterVector>::AttributeAdapter(ParameterVector& ref)
    : m_ref(ref)
{
}

bool AttributeAdapter<ParameterVector>::visit_attributes(AttributeVisitor& visitor)
{
    int64_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size())
    {
        m_ref.resize(size);
    }
    ostringstream index;
    for (int64_t i = 0; i < size; i++)
    {
        index.str("");
        index << i;
        string id;
        if (m_ref[i])
        {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i])
        {
            m_ref[i] = as_type_ptr<op::v0::Parameter>(visitor.get_registered_node(id));
        }
    }
    return true;
}
