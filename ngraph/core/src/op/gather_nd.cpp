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

#include "ngraph/op/gather_nd.hpp"
#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V5 ------------------------------

NGRAPH_RTTI_DEFINITION(op::v5::GatherND, "GatherND", 5);

op::v5::GatherND::GatherND(const Output<Node>& data,
                           const Output<Node>& indices,
                           const size_t batch_dims)
    : Op({data, indices})
    , m_batch_dims(batch_dims)
{
    constructor_validate_and_infer_types();
}

void op::v5::GatherND::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v5_GatherND_validate_and_infer_types);
    // check types of input tensors
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);

    // check ranks of input tensors
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);

    if (data_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this, data_pshape.rank().get_length() > 0, "Data rank must be at least 1.");

        NODE_VALIDATION_CHECK(this,
                              data_pshape.rank().get_length() > m_batch_dims,
                              "Number of batch dimensions must not exceed a rank of data.");
    }

    if (indices_pshape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this, indices_pshape.rank().get_length() > 0, "Indices rank must be at least 1.");

        NODE_VALIDATION_CHECK(this,
                              indices_pshape.rank().get_length() > m_batch_dims,
                              "Number of batch dimensions must not exceed a rank of indices.");
    }

    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static())
    {
        // check that batch dimensions of data and indices are the same
        for (auto batch_dim = 0; batch_dim < m_batch_dims; batch_dim++)
        {
            if (data_pshape[batch_dim].is_static() && indices_pshape[batch_dim].is_static())
            {
                NODE_VALIDATION_CHECK(this,
                                      data_pshape[batch_dim].get_length() ==
                                          indices_pshape[batch_dim].get_length(),
                                      "Batch dimensions of data and indices must be the same.");
            }
        }

        if (indices_pshape[indices_pshape.rank().get_length() - 1].is_static())
        {
            NODE_VALIDATION_CHECK(
                this,
                (indices_pshape[indices_pshape.rank().get_length() - 1].get_length() +
                 m_batch_dims) <= data_pshape.rank().get_length(),
                "Length of a tuple with indices must not exceed a rank of data tensor "
                "excluding "
                "batch dimensions.");
        }
    }

    // set output shape
    set_output_size(1);
    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static() &&
        indices_pshape[indices_pshape.rank().get_length() - 1].is_static())
    {
        auto indices_tuple_length =
            indices_pshape[indices_pshape.rank().get_length() - 1].get_length();
        auto slice_length = data_pshape.rank().get_length() - indices_tuple_length - m_batch_dims;
        auto output_indices_length = indices_pshape.rank().get_length() - m_batch_dims - 1;
        auto output_rank = output_indices_length + slice_length;
        size_t delta_output_rank = 0;
        if (m_batch_dims > 0)
        {
            delta_output_rank = 1;
        }
        std::vector<Dimension> output_shape(output_rank + delta_output_rank);
        if (m_batch_dims > 0)
        {
            output_shape[0] = 1;
            for (auto dim = 0; dim < m_batch_dims; dim++)
            {
                if (data_pshape[dim].is_static())
                {
                    output_shape[0] *= data_pshape[dim].get_length();
                }
                else if (indices_pshape[dim].is_static())
                {
                    output_shape[0] *= indices_pshape[dim].get_length();
                }
                else
                {
                    output_shape[0] = Dimension::dynamic();
                    break;
                }
            }
        }
        for (auto dim = 0; dim < output_indices_length; dim++)
        {
            output_shape[dim + delta_output_rank] = indices_pshape[dim + m_batch_dims];
        }
        for (auto dim = 0; dim < slice_length; dim++)
        {
            output_shape[output_indices_length + dim + delta_output_rank] =
                data_pshape[m_batch_dims + indices_tuple_length + dim];
        }
        set_output_type(0, data_type, PartialShape(output_shape));
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

bool op::v5::GatherND::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v5_GatherND_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

shared_ptr<Node> op::v5::GatherND::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v5_GatherND_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v5::GatherND>(new_args.at(0), new_args.at(1), m_batch_dims);
}
