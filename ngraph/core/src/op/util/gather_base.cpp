// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/gather_base.hpp"
#include "itt.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/shape.hpp"

#include <ngraph/validation_util.hpp>

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::GatherBase, "GatherBase", 7);

op::util::GatherBase::GatherBase(const Output<Node>& data,
                                 const Output<Node>& indices,
                                 const Output<Node>& axis,
                                 const int64_t batch_dims)
    : Op({data, indices, axis})
    , m_batch_dims(batch_dims)
{
    constructor_validate_and_infer_types();
}

void op::util::GatherBase::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_GatherBase_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    const auto& axis_pshape = get_input_partial_shape(2);
    auto data_rank = data_pshape.rank();
    auto indices_rank = indices_pshape.rank();
    auto axis_rank = axis_pshape.rank();

    if (axis_rank.is_static() && axis_pshape.is_static())
    {
        const auto axis_is_scalar = axis_rank.get_length() == 0;
        const auto axis_has_one_elem =
            axis_rank.get_length() == 1 && axis_pshape[0].get_length() == 1;
        NODE_VALIDATION_CHECK(
            this,
            axis_is_scalar || axis_has_one_elem,
            "Axis input must be scalar or have 1 element. But instead got axis_shape = ",
            axis_pshape);
    }

    int64_t batch_dims = m_batch_dims;
    if (batch_dims < 0 && indices_rank.is_static())
    {
        batch_dims += indices_rank.get_length();
    }

    bool axis_is_set = false;
    if (get_constant_from_source(input_value(2)))
        axis_is_set = true;

    if (axis_is_set)
    {
        int64_t axis = get_axis(); // will be normalized to positive if data_rank is static

        // batch_dims, axis both can be positive by default or after normalization if data_rank &
        // indices_rank are static.
        // If at least one of them is negative we cannot check their consistency.
        NODE_VALIDATION_CHECK(
            this,
            batch_dims <= axis || batch_dims < 0 || axis < 0,
            "After normalization batch_dims must be <= axis. But instead got: batch_dims = ",
            batch_dims,
            ", axis = ",
            axis);

        NODE_VALIDATION_CHECK(
            this,
            data_rank.is_dynamic() || (axis >= 0 && axis < data_rank.get_length()),
            "Normalized axis must be >= 0 and < data_rank. But instead got axis = ",
            axis,
            " data_rank = ",
            data_rank.get_interval());
    }

    if (indices_rank.is_static() && batch_dims >= 0)
    {
        NODE_VALIDATION_CHECK(
            this,
            batch_dims <= indices_rank.get_length(),
            "The batch_dims must be <= indices_rank. But instead got: batch_dims = ",
            batch_dims,
            ", indices_rank = ",
            indices_rank.get_length());
    }

    if (data_rank.is_static() && indices_rank.is_static())
    {
        auto out_rank = data_rank.get_length() + indices_rank.get_length() - 1 - batch_dims;
        PartialShape output_pshape = PartialShape::dynamic(out_rank);

        // implementation of out_shape formula
        // data.shape[:batch_dims] + data.shape[batch_dims:axis] + indices.shape[batch_dims:] +
        // data.shape[axis + 1:]
        int i = 0;
        for (; i < batch_dims; i++)
        {
            NODE_VALIDATION_CHECK(this,
                                  data_pshape[i].compatible(indices_pshape[i]),
                                  "Shapes ",
                                  data_pshape,
                                  " and ",
                                  indices_pshape,
                                  " are not consistent. data and indices must have equal or "
                                  "intersecting sizes until batch_dims");

            output_pshape[i] = data_pshape[i] & indices_pshape[i];
        }

        if (axis_is_set)
        {
            int64_t axis = get_axis();
            for (; i < axis; i++)
            {
                output_pshape[i] = data_pshape[i];
            }
            for (; i < axis + indices_rank.get_length() - batch_dims; i++)
            {
                output_pshape[i] = indices_pshape[batch_dims - axis + i];
            }
            for (; i < out_rank; i++)
            {
                output_pshape[i] = data_pshape[batch_dims + 1 - indices_rank.get_length() + i];
            }
        }
        set_output_type(0, data_type, output_pshape);
    }
    else
    {
        Rank out_rank = data_rank + indices_rank - 1 - batch_dims;
        if (batch_dims < 0)
            out_rank = out_rank - indices_rank.get_max_length();
        set_output_type(0, data_type, PartialShape::dynamic(out_rank));
    }
}

int64_t op::util::GatherBase::get_axis() const
{
    const auto& const_op = get_constant_from_source(input_value(2));
    if (!const_op)
        throw ngraph_error("axis value is not set");

    int64_t axis = const_op->cast_vector<int64_t>()[0];
    if (axis < 0)
    {
        const auto& data_rank = get_input_partial_shape(0).rank();
        if (data_rank.is_static())
        {
            axis += data_rank.get_length();
        }
    }
    return axis;
}

namespace gather
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  int64_t axis,
                  int64_t batch_dims)
    {
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = arg0->get_shape();
        Shape indices_shape = arg1->get_shape();
        Shape out_shape(params_shape.size() + indices_shape.size() - 1 - batch_dims);
        int64_t i = 0;
        for (; i < axis; i++)
        {
            out_shape[i] = params_shape[i];
        }
        for (int64_t j = batch_dims; j < static_cast<int64_t>(indices_shape.size()); i++, j++)
        {
            out_shape[i] = indices_shape[j];
        }
        for (int64_t j = axis + 1; j < static_cast<int64_t>(params_shape.size()); i++, j++)
        {
            out_shape[i] = params_shape[j];
        }

        out->set_shape(out_shape);

        if (arg1->get_element_type() == element::i64)
        {
            runtime::reference::gather<T, int64_t>(arg0->get_data_ptr<ET>(),
                                                   arg1->get_data_ptr<int64_t>(),
                                                   out->get_data_ptr<ET>(),
                                                   arg0->get_shape(),
                                                   arg1->get_shape(),
                                                   out->get_shape(),
                                                   axis,
                                                   batch_dims);
        }
        else if (arg1->get_element_type() == element::i32)
        {
            runtime::reference::gather<T, int32_t>(arg0->get_data_ptr<ET>(),
                                                   arg1->get_data_ptr<int32_t>(),
                                                   out->get_data_ptr<ET>(),
                                                   arg0->get_shape(),
                                                   arg1->get_shape(),
                                                   out->get_shape(),
                                                   axis,
                                                   batch_dims);
        }
        else
        {
            throw ngraph_error("Unexpected type");
        }

        return true;
    }

    bool evaluate_gather(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& out,
                         int64_t axis,
                         int64_t batch_dims = 0)
    {
        bool rc = true;

        switch (out->get_element_type())
        {
            NGRAPH_TYPE_CASE(evaluate_gather, i32, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, i64, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, u32, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, u64, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, f16, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, f32, arg0, arg1, out, axis, batch_dims);
            NGRAPH_TYPE_CASE(evaluate_gather, boolean, arg0, arg1, out, axis, batch_dims);
        default: rc = false; break;
        }
        return rc;
    }

    bool cf_gather_with_subgraph(OutputVector& output_values,
                                 const OutputVector& input_values,
                                 const PartialShape& gather_ps)
    {
        if (gather_ps.is_dynamic() || input_values.size() != 3)
        {
            return false;
        }

        const auto concat =
            std::dynamic_pointer_cast<op::Concat>(input_values[0].get_node_shared_ptr());
        const auto indices =
            std::dynamic_pointer_cast<op::Constant>(input_values[1].get_node_shared_ptr());
        const auto axis =
            std::dynamic_pointer_cast<op::Constant>(input_values[2].get_node_shared_ptr());

        if (!concat || !indices || !axis)
        {
            return false;
        }

        // only along axis=0
        if (axis->cast_vector<int64_t>()[0] != 0 || concat->get_axis() != 0)
        {
            return false;
        }
        // only single indices are accepted
        const auto indices_shape = indices->get_shape();
        if (indices_shape.size() > 1 || (indices_shape.size() == 1 && indices_shape[0] > 1))
        {
            return false;
        }
        // concat inputs are 1D and their count is equal to Concat output shape
        if (concat->get_output_partial_shape(0).is_dynamic())
        {
            return false;
        }
        const auto concat_inputs = concat->inputs();
        // concat inputs must be single elements
        if (concat_inputs.size() != shape_size(concat->get_shape()))
        {
            return false;
        }

        const int64_t rank = concat->get_shape()[0];
        const int64_t raw_index = indices->cast_vector<int64_t>()[0];
        const int64_t positive_index = raw_index < 0 ? rank + raw_index : raw_index;
        NGRAPH_CHECK(positive_index >= 0 && positive_index < rank);

        // gather takes exactly one element out of the Concat output
        const auto gathered_concat_input =
            concat_inputs[positive_index].get_source_output().get_node_shared_ptr();
        // Concat inputs are 1D, resulting tensor shape depends on Gather indices
        auto gathered = gathered_concat_input;
        if (indices_shape.empty())
        {
            // gathering a scalar
            const auto axis_const = op::Constant::create(element::i64, Shape{1}, {0});
            gathered = make_shared<op::v0::Squeeze>(gathered_concat_input, axis_const);
        }

        output_values[0] = gathered;

        return true;
    }
} // namespace gather

bool op::util::GatherBase::evaluate(const HostTensorVector& outputs,
                                    const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(util_GatherBase_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(inputs, 3));
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1));

    int64_t axis = 0;
    switch (inputs[2]->get_element_type())
    {
    case element::Type_t::i32: axis = inputs[2]->get_data_ptr<element::Type_t::i32>()[0]; break;
    case element::Type_t::i64: axis = inputs[2]->get_data_ptr<element::Type_t::i64>()[0]; break;
    case element::Type_t::i8: axis = inputs[2]->get_data_ptr<element::Type_t::i8>()[0]; break;
    case element::Type_t::i16: axis = inputs[2]->get_data_ptr<element::Type_t::i16>()[0]; break;
    case element::Type_t::u8: axis = inputs[2]->get_data_ptr<element::Type_t::u8>()[0]; break;
    case element::Type_t::u16: axis = inputs[2]->get_data_ptr<element::Type_t::u16>()[0]; break;
    case element::Type_t::u32: axis = inputs[2]->get_data_ptr<element::Type_t::u32>()[0]; break;
    case element::Type_t::u64: axis = inputs[2]->get_data_ptr<element::Type_t::u64>()[0]; break;
    default: throw ngraph_error("axis must be of integral data type.");
    }

    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(0).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }

    int64_t batch_dims = m_batch_dims;
    const auto& indices_rank = get_input_partial_shape(1).rank();
    if (batch_dims < 0 && indices_rank.is_static())
        batch_dims += indices_rank.get_length();

    return gather::evaluate_gather(inputs[0], inputs[1], outputs[0], axis, batch_dims);
}

bool op::util::GatherBase::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::util::GatherBase::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::util::GatherBase::constant_fold(OutputVector& output_values,
                                         const OutputVector& input_values)
{
    // try the regular constant folding just for the Gather node
    if (Node::constant_fold(output_values, input_values))
    {
        return true;
    }
    else
    {
        return gather::cf_gather_with_subgraph(
            output_values, input_values, get_output_partial_shape(0));
    }
}
