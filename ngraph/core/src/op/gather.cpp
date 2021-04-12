// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather.hpp"
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

constexpr NodeTypeInfo op::v1::Gather::type_info;
const int64_t op::v1::Gather::AXIS_NOT_SET_VALUE;

const int op::v1::Gather::PARAMS = 0;
const int op::v1::Gather::INDICES = 1;
const int op::v1::Gather::AXIS = 2;

op::v1::Gather::Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axes)
    : Op({params, indices, axes})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Gather::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Gather_visit_attributes);
    return true;
}

void op::v1::Gather::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Gather_validate_and_infer_types);
    const auto& input_rank = get_input_partial_shape(PARAMS).rank();
    const auto& axis_shape = get_input_partial_shape(AXIS);
    const auto& axis_rank = axis_shape.rank();

    if (axis_rank.is_static() && axis_shape.is_static())
    {
        const auto axis_is_scalar = axis_rank.get_length() == 0;
        const auto axis_has_one_elem =
            axis_rank.get_length() == 1 && axis_shape[0].get_length() == 1;
        NODE_VALIDATION_CHECK(this,
                              axis_is_scalar || axis_has_one_elem,
                              "Axes input must be scalar or have 1 element (shape: ",
                              axis_shape,
                              ").");
    }

    int64_t axis = get_axis();
    if (input_rank.is_static() && axis != AXIS_NOT_SET_VALUE)
    {
        NODE_VALIDATION_CHECK(this,
                              axis < input_rank.get_length(),
                              "The axis must => 0 and <= input_rank (axis: ",
                              axis,
                              ").");
    }

    element::Type result_et = get_input_element_type(PARAMS);

    const PartialShape& params_shape = get_input_partial_shape(PARAMS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    PartialShape result_shape;
    if (params_shape.rank().is_static() && indices_shape.rank().is_static() &&
        axis != AXIS_NOT_SET_VALUE)
    {
        std::vector<Dimension> result_dims(params_shape.rank().get_length() +
                                           indices_shape.rank().get_length() - 1);
        uint64_t i = 0;
        for (; i < axis; i++)
        {
            result_dims[i] = params_shape[i];
        }
        for (uint64_t j = 0; j < indices_shape.rank().get_length(); i++, j++)
        {
            result_dims[i] = indices_shape[j];
        }
        for (uint64_t j = axis + 1; j < params_shape.rank().get_length(); i++, j++)
        {
            result_dims[i] = params_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

int64_t op::v1::Gather::get_axis() const
{
    int64_t axis = AXIS_NOT_SET_VALUE;
    if (const auto& const_op = get_constant_from_source(input_value(AXIS)))
    {
        axis = const_op->cast_vector<int64_t>()[0];
    }
    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(PARAMS).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }
    return axis;
}

shared_ptr<Node> op::v1::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Gather>(new_args.at(PARAMS), new_args.at(INDICES), new_args.at(AXIS));
}

NGRAPH_RTTI_DEFINITION(op::v7::Gather, "Gather", 7);

op::v7::Gather::Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims)
    : Op({data, indices, axis})
    , m_batch_dims(batch_dims)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v7::Gather::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

void op::v7::Gather::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_Gather_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type == element::Type_t::i32 ||
                              indices_type == element::Type_t::i64,
                          "indices must be of int32 or int64 type. But instead got: ",
                          indices_type);

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
            "Axes input must be scalar or have 1 element. But instead got axis_shape = ",
            axis_pshape);
    }

    int64_t batch_dims = get_batch_dims(); // will not be converted to positive if axis is not set
    if (is_axis_set())
    {
        int64_t axis = get_axis();
        NODE_VALIDATION_CHECK(this,
                              batch_dims <= axis,
                              "The batch_dims <= axis. But instead got: batch_dims = ",
                              batch_dims,
                              ", axis = ",
                              axis);

        if (data_rank.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  axis >= 0 && axis < data_rank.get_length(),
                                  "The axis must be => 0 and < data_rank. But instead got axis = ",
                                  axis,
                                  " data_rank = ",
                                  data_rank.get_length());
        }
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
        if (batch_dims >= 0)
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

            if (is_axis_set())
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
        else if (batch_dims < 0)
        {
            // batch_dims < 0 could be only if axis is not set
            // as soon as axis value will arrive negative batch_dims should be resolved
            // batch_dims value will be within [0, data_rank] && [0, indices_rank]
            int64_t max_rank = data_rank.get_length() + indices_rank.get_length() - 1;
            int64_t min_rank = max_rank - max(data_rank.get_length(), indices_rank.get_length());

            set_output_type(0, data_type, PartialShape::dynamic(Dimension(min_rank, max_rank)));
        }
    }
    else
    {
        set_output_type(0, data_type, PartialShape::dynamic());
    }
}

int64_t op::v7::Gather::get_axis() const
{
    const auto& const_op = get_constant_from_source(input_value(2));
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

int64_t op::v7::Gather::get_batch_dims() const
{
    if (m_batch_dims < 0 && is_axis_set())
        return get_axis() + m_batch_dims;
    else
        return m_batch_dims;
}

bool op::v7::Gather::is_axis_set() const
{
    const auto& axes_constant = get_constant_from_source(input_value(2));
    if (axes_constant)
        return true;
    else
        return false;
}

shared_ptr<Node> op::v7::Gather::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}

namespace gather
{
    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& out,
                  size_t axis,
                  size_t batch_dims)
    {
        using T = typename element_type_traits<ET>::value_type;
        Shape params_shape = arg0->get_shape();
        Shape indices_shape = arg1->get_shape();
        Shape out_shape(params_shape.size() + indices_shape.size() - 1 - batch_dims);
        uint64_t i = 0;
        for (; i < axis; i++)
        {
            out_shape[i] = params_shape[i];
        }
        for (uint64_t j = batch_dims; j < indices_shape.size(); i++, j++)
        {
            out_shape[i] = indices_shape[j];
        }
        for (uint64_t j = axis + 1; j < params_shape.size(); i++, j++)
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
                         size_t axis,
                         size_t batch_dims = 0)
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
            const auto axes = op::Constant::create(element::i64, Shape{1}, {0});
            gathered = make_shared<op::v0::Squeeze>(gathered_concat_input, axes);
        }

        output_values[0] = gathered;

        return true;
    }
} // namespace gather

bool op::v1::Gather::evaluate_gather(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    int64_t axis = 0;
    switch (inputs[2]->get_element_type())
    {
    case element::Type_t::i8: axis = inputs[2]->get_data_ptr<element::Type_t::i8>()[0]; break;
    case element::Type_t::i16: axis = inputs[2]->get_data_ptr<element::Type_t::i16>()[0]; break;
    case element::Type_t::i32: axis = inputs[2]->get_data_ptr<element::Type_t::i32>()[0]; break;
    case element::Type_t::i64: axis = inputs[2]->get_data_ptr<element::Type_t::i64>()[0]; break;
    case element::Type_t::u8: axis = inputs[2]->get_data_ptr<element::Type_t::u8>()[0]; break;
    case element::Type_t::u16: axis = inputs[2]->get_data_ptr<element::Type_t::u16>()[0]; break;
    case element::Type_t::u32: axis = inputs[2]->get_data_ptr<element::Type_t::u32>()[0]; break;
    case element::Type_t::u64: axis = inputs[2]->get_data_ptr<element::Type_t::u64>()[0]; break;
    default: throw ngraph_error("axis element type is not integral data type");
    }

    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(PARAMS).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }
    return gather::evaluate_gather(inputs[0], inputs[1], outputs[0], axis);
}

bool op::v1::Gather::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Gather_evaluate);
    NGRAPH_CHECK(this, validate_host_tensor_vector(inputs, 3));
    NGRAPH_CHECK(this, validate_host_tensor_vector(outputs, 1));
    return evaluate_gather(outputs, inputs);
}

bool op::v1::Gather::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(INDICES).get_tensor().has_and_set_bound() ||
        !input_value(AXIS).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Gather::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(INDICES).get_tensor().has_and_set_bound() ||
        !input_value(AXIS).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v1::Gather::constant_fold(OutputVector& output_values, const OutputVector& input_values)
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

bool op::v7::Gather::evaluate_gather(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    int64_t axis = 0;
    switch (inputs[2]->get_element_type())
    {
    case element::Type_t::i32: axis = inputs[2]->get_data_ptr<element::Type_t::i32>()[0]; break;
    case element::Type_t::i64: axis = inputs[2]->get_data_ptr<element::Type_t::i64>()[0]; break;
    default: throw ngraph_error("axis must be of int32 or int64 type.");
    }

    if (axis < 0)
    {
        const auto& input_rank = get_input_partial_shape(0).rank();
        if (input_rank.is_static())
        {
            axis += input_rank.get_length();
        }
    }
    return gather::evaluate_gather(inputs[0], inputs[1], outputs[0], axis, get_batch_dims());
}

bool op::v7::Gather::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v7_Gather_evaluate);
    NGRAPH_CHECK(this, validate_host_tensor_vector(inputs, 3));
    NGRAPH_CHECK(this, validate_host_tensor_vector(outputs, 1));
    return evaluate_gather(outputs, inputs);
}

bool op::v7::Gather::evaluate_lower(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool op::v7::Gather::evaluate_upper(const HostTensorVector& output_values) const
{
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        !input_value(2).get_tensor().has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool op::v7::Gather::constant_fold(OutputVector& output_values, const OutputVector& input_values)
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
