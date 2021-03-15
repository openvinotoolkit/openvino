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

#include <memory>
#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/index_reduction.hpp"

using namespace std;
using namespace ngraph;

op::util::IndexReduction::IndexReduction() {}

op::util::IndexReduction::IndexReduction(const Output<Node>& arg,
                                         uint64_t axis,
                                         const element::Type& index_element_type)
    : Op({arg})
{
    set_reduction_axis(axis);
    set_index_element_type(index_element_type);
}

uint64_t op::util::IndexReduction::get_reduction_axis() const
{
    return m_axis;
}
void op::util::IndexReduction::set_reduction_axis(uint64_t value)
{
    m_axis = value;
}
element::Type op::util::IndexReduction::get_index_element_type() const
{
    return m_index_element_type;
}
void op::util::IndexReduction::set_index_element_type(const element::Type& index_element_type)
{
    m_index_element_type = index_element_type;
}

void op::util::IndexReduction::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_IndexReduction_validate_and_infer_types);
    // TODO(amprocte): Should reject if size of reduction axis is zero.
    const PartialShape& arg_shape = get_input_partial_shape(0);
    Rank rank = arg_shape.rank();

    NODE_VALIDATION_CHECK(
        this, rank.is_dynamic() || rank.get_length() >= 1, "Argument rank is zero.");
    NODE_VALIDATION_CHECK(this,
                          rank.is_dynamic() || m_axis < rank.get_length(),
                          "Reduction axis (",
                          m_axis,
                          ") is not less than argument rank (",
                          rank,
                          ").");
    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i32 ||
                              m_index_element_type == element::i64,
                          "Index element is neither i64 or i32.");

    PartialShape output_shape{PartialShape::dynamic()};

    if (rank.is_static())
    {
        Dimension d = arg_shape[m_axis];
        if (d.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  0 != d.get_length(),
                                  "Tensor reduction axis can not be empty, shape is: ",
                                  arg_shape);
        }

        std::vector<Dimension> output_dims(rank.get_length() - 1);
        size_t j = 0;

        for (size_t i = 0; i < rank.get_length() - 1; i++)
        {
            if (j == m_axis)
            {
                j++;
            }
            output_dims[i] = arg_shape[j++];
        }

        output_shape = PartialShape(output_dims);
    }

    set_output_type(0, m_index_element_type, output_shape);
}

bool op::util::IndexReduction::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(util_IndexReduction_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}
