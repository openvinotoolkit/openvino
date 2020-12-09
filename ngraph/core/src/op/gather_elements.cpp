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
                                       const int64_t axis)
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
                          indices_type == element::Type_t::i32 ||
                              indices_type == element::Type_t::i64,
                          "indices mush be of int32 or int64 type. But instead got: ",
                          indices_type);

    // check ranks of input tensors
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();

    if (data_rank.is_static())
    {
        int64_t data_rank_size = data_rank.get_length();

        NODE_VALIDATION_CHECK(this, data_rank_size >= 1, "data rank must be >= 1.");

        if (m_axis < 0)
        {
            NODE_VALIDATION_CHECK(
                this,
                (-data_rank_size < m_axis) && (m_axis < data_rank_size - 1),
                "axis must be within interval (-data.rank,  data.rank - 1). But instead Got: ",
                m_axis);
            m_axis += data_rank_size;
        }
    }

    if (indices_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this, indices_rank.get_length() >= 1, "indices rank must be >= 1.");
    }

    if (indices_rank.is_static() && data_rank.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              data_rank.get_length() == indices_rank.get_length(),
                              "data and indices rank must be equal. But instead got: ",
                              data_rank.get_length(),
                              " and ",
                              indices_rank.get_length());

        PartialShape out_shape_info(indices_pshape);
        stringstream data_shape_val, indices_shape_val;
        for (int i = 0; i < indices_rank.get_length(); i++)
        {
            if (i != m_axis)
            {
                // if size of the current indices dim is unknown it will retrieve it from data
                // e.g., if data_shape = {4, 4, ?} indices_shape = {1, ?, 5} and axis = 0 (and if
                // intervals intersect) then out size along dim 2 is also 5
                Dimension curr_dim = data_pshape[i] & indices_pshape[i];

                if (data_pshape.is_static())
                    data_shape_val << data_pshape[i];
                else
                    data_shape_val << data_pshape[i].get_interval();

                if (indices_pshape.is_static())
                    indices_shape_val << indices_pshape[i];
                else
                    indices_shape_val << indices_pshape[i].get_interval();

                // in the static case indices and data must have the same sizes along all dims
                // except axis in the dynamic case intervals must intersect
                NODE_VALIDATION_CHECK(this,
                                      !curr_dim.get_interval().empty(),
                                      "Sizes ",
                                      data_shape_val.str(),
                                      " and ",
                                      indices_shape_val.str(),
                                      " on axis ",
                                      i,
                                      " do not match. data and indices must have equal or "
                                      "intersecting shapes except for axis ",
                                      m_axis);

                out_shape_info[i] = curr_dim;
            }
        }
        set_output_type(0, data_type, out_shape_info);
    }
    else // if (indices_rank.is_dynamic() || data_rank.is_dynamic())
    {
        // if at least one input has static rank propagate PartialShape of that input
        // in the optimistic scenario at least we will have static rank
        // in the worse scenario propagating any of PartialShapes is equivalent
        set_output_type(0, data_type, indices_pshape.is_static() ? indices_pshape : data_pshape);
    }
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
