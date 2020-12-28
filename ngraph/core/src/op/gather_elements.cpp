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
#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V6 ------------------------------

NGRAPH_RTTI_DEFINITION(op::v6::GatherElements, "GatherElements", 6);

op::v6::GatherElements::GatherElements(const Output<Node>& data,
                                       const Output<Node>& indices,
                                       const int64_t axis)
    : Op({data, indices})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

void op::v6::GatherElements::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_GatherElements_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type == element::Type_t::i32 ||
                              indices_type == element::Type_t::i64,
                          "indices must be of int32 or int64 type. But instead got: ",
                          indices_type);

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();

    int64_t axis = m_axis;
    if (m_axis < 0 && data_rank.is_static())
        axis += data_rank.get_length();

    set_output_type(0, data_type, indices_pshape);

    NODE_VALIDATION_CHECK(
        this, data_rank.is_dynamic() || data_rank.get_length() >= 1, "data rank must be >= 1.");

    NODE_VALIDATION_CHECK(
        this,
        data_rank.is_dynamic() ||
            ((-data_rank.get_length() <= m_axis) && (m_axis < data_rank.get_length())),
        "axis must be within interval (-data.rank,  data.rank - 1). But instead Got: ",
        m_axis);

    NODE_VALIDATION_CHECK(this,
                          indices_rank.is_dynamic() || indices_rank.get_length() >= 1,
                          "indices rank must be >= 1.");

    if (data_rank.is_static() && indices_rank.is_dynamic())
    {
        PartialShape out_shape_info(data_pshape);
        out_shape_info[axis] = Dimension::dynamic();
        set_output_type(0, data_type, out_shape_info);
        return;
    }

    if (data_rank.is_dynamic())
    {
        if (indices_rank.is_dynamic())
            set_output_type(0, data_type, PartialShape::dynamic());
        return;
    }

    // left only case when data_rank.is_static() && indices_rank.is_static()
    NODE_VALIDATION_CHECK(this,
                          data_rank.get_length() == indices_rank.get_length(),
                          "data and indices rank must be equal. But instead got: ",
                          data_rank.get_length(),
                          " and ",
                          indices_rank.get_length());

    PartialShape output_pshape(indices_pshape);
    for (int i = 0; i < indices_rank.get_length(); i++)
    {
        if (i != axis)
        {
            // if size of the current axis of indices is unknown it will retrieve it from data
            // e.g., if data_shape = {4, 4, ?} indices_shape = {1, ?, 5} and axis = 0
            // (and if intervals intersect) then output_pshape will be {1, 4, 5}
            Dimension curr_dim = data_pshape[i] & indices_pshape[i];

            NODE_VALIDATION_CHECK(this,
                                  !curr_dim.get_interval().empty(),
                                  "Shapes ",
                                  data_pshape,
                                  " and ",
                                  indices_pshape,
                                  " are not consistent. data and indices must have equal or "
                                  "intersecting sizes, except for axis ",
                                  m_axis);

            output_pshape[i] = curr_dim;
        }
    }
    set_output_type(0, data_type, output_pshape);
}

bool op::v6::GatherElements::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_GatherElements_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v6::GatherElements::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_GatherElements_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::GatherElements>(new_args.at(0), new_args.at(1), m_axis);
}
