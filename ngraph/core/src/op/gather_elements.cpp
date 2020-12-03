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

#include "ngraph/op/gather_elements.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V6 ------------------------------

NGRAPH_RTTI_DEFINITION(op::v6::GatherElements, "GatherElements", 6);

op::v6::GatherElements::GatherElements(const Output<Node>& data,
                                       const Output<Node>& indices,
                                       const size_t axis)
    : Op({data, indices})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

void op::v6::GatherElements::validate_and_infer_types()
{
    // check types of input tensors
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type == element::Type_t::i32 || indices_type == element::Type_t::i64,
                          "indices mush be of int32 or int64 type. But instead got: ",
                          indices_type);

    // check ranks of input tensors
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();

    if (data_rank.is_static())
    {
        auto data_rank_size = data_rank.get_length();

        NODE_VALIDATION_CHECK(this, data_rank_size > 1, "Data rank must be greater than 1.");

        if (m_axis < 0)
        {
            NODE_VALIDATION_CHECK(
                this,
                -data_rank_size < m_axis < data_rank_size - 1,
                "axis must be within interval (-data.rank,  data.rank - 1. But instead Got: ",
                m_axis);
            m_axis = data_rank_size + m_axis;
        }
    }

    if (indices_rank.is_static())
    {
        NODE_VALIDATION_CHECK(
            this, indices_rank.get_length() > 1, "Indices rank must be greater that 1.");
    }

    if (data_rank.is_static() && indices_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              data_rank.get_length() == indices_rank.get_length(),
                              "data and indices rank must be equal. But instead got: ",
                              data_rank.get_length(),
                              " and ",
                              indices_rank.get_length());

        if (data_pshape.is_static() && indices_pshape.is_static())
        {
            // check if PartialShapes of data and indices are consistent
            for (int i = 0; i < data_rank.get_length(); i++)
            {
                if (i != m_axis)
                    NODE_VALIDATION_CHECK(
                        this,
                        data_pshape[i] == indices_pshape[i],
                        "Sizes ",
                        data_pshape[i],
                        " and ",
                        indices_pshape[i],
                        " on axis ",
                        i,
                        " do not match. data and indices mush have equal shapes except for axis ",
                        m_axis);
            }
        }
    }

    set_output_type(0, data_type, indices_pshape);
}

bool op::v6::GatherElements::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v6::GatherElements::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v6::GatherElements>(new_args.at(0), new_args.at(1), m_axis);
}
