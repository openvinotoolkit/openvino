// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/slice.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v1::VariadicSplit, "VariadicSplit", 1);

op::v1::VariadicSplit::VariadicSplit(const Output<Node>& data,
                                     const Output<Node>& axis,
                                     const Output<Node>& split_lengths)
    : Op({data, axis, split_lengths})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::VariadicSplit::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_VariadicSplit_visit_attributes);
    return true;
}

void ngraph::op::v1::VariadicSplit::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_VariadicSplit_validate_and_infer_types);
    set_input_is_relevant_to_value(0);
    set_input_is_relevant_to_value(1);
    set_input_is_relevant_to_value(2);

    auto split_lengths_pshape = get_input_partial_shape(2);

    if (split_lengths_pshape.is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              split_lengths_pshape.rank().get_length() == 1,
                              "Split lengths should be a 1-D tensor. Got ",
                              split_lengths_pshape.rank(),
                              " instead.");

        auto num_outputs = split_lengths_pshape[0].get_length();
        auto data = input_value(0);
        auto axis_source = input_value(1);
        auto split_lengths_source = input_value(2);
        auto data_shape = data.get_partial_shape();
        const auto& data_type = data.get_element_type();

        set_output_size(num_outputs);
        const auto& axis_input_constant = get_constant_from_source(axis_source);
        const auto& split_lengths_constant = get_constant_from_source(split_lengths_source);
        if (data_shape.rank().is_static() && axis_input_constant && split_lengths_constant)
        {
            auto axis_val = axis_input_constant->cast_vector<int64_t>()[0];
            // Adjust split axis in case of negatives
            int64_t axis = ngraph::normalize_axis(this, axis_val, data_shape.rank());

            auto split_lengths = split_lengths_constant->cast_vector<int64_t>();
            // Adjust split lengths in case of negatives
            int64_t sum_of_splits = 0;
            int64_t negative_one = -1;
            for (size_t i = 0; i < split_lengths.size(); i++)
            {
                NODE_VALIDATION_CHECK(this,
                                      split_lengths[i] >= -1,
                                      "Invalid value ",
                                      split_lengths[i],
                                      " in split lengths input. Should be >= -1.");

                if (split_lengths[i] == -1)
                {
                    NODE_VALIDATION_CHECK(this,
                                          negative_one == -1,
                                          "Cannot infer split with multiple -1 values at ",
                                          negative_one,
                                          " and ",
                                          i);
                    negative_one = i;
                }
                else
                {
                    sum_of_splits += split_lengths[i];
                }
            }
            auto data_shape_dims = vector<Dimension>{data.get_partial_shape()};
            auto dimension_at_axis = data_shape_dims.at(axis);

            if (negative_one >= 0 && dimension_at_axis.is_static())
            {
                split_lengths[negative_one] = dimension_at_axis.get_length() - sum_of_splits;
                sum_of_splits += split_lengths[negative_one];
            }
            if (data_shape[axis].is_static())
            {
                NODE_VALIDATION_CHECK(this,
                                      sum_of_splits == data_shape[axis].get_length(),
                                      "Total length of splits: ",
                                      sum_of_splits,
                                      " must match the length of the chosen axis: ",
                                      data_shape[axis]);
            }

            for (int64_t output{0}; output < num_outputs; ++output)
            {
                auto output_split_dim = split_lengths.at(output) == -1 ? Dimension::dynamic()
                                                                       : split_lengths.at(output);
                auto tmp_shape = data_shape_dims;
                tmp_shape.at(axis) = output_split_dim;
                set_output_type(output, data_type, PartialShape{tmp_shape});
            }
        }
        else
        {
            for (int64_t output{0}; output < num_outputs; ++output)
            {
                set_output_type(output, data_type, PartialShape::dynamic());
            }
        }
    }
}

shared_ptr<Node> op::v1::VariadicSplit::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_VariadicSplit_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::VariadicSplit>(new_args.at(0), new_args.at(1), new_args.at(2));
}

namespace variadic_split
{
    inline bool evaluate(const HostTensorPtr& in,
                         const HostTensorPtr& out,
                         const Coordinate& lower_bounds,
                         const Coordinate& upper_bounds)
    {
        runtime::reference::slice(in->get_data_ptr<const char>(),
                                  out->get_data_ptr<char>(),
                                  in->get_shape(),
                                  lower_bounds,
                                  upper_bounds,
                                  Strides(lower_bounds.size(), 1),
                                  out->get_shape(),
                                  in->get_element_type().size());

        return true;
    }
} // namespace variadic_split

bool op::v1::VariadicSplit::evaluate_variadic_split(const HostTensorVector& inputs,
                                                    const HostTensorVector& outputs) const
{
    const auto& data_tensor = inputs[0];
    const auto& axis_tensor = inputs[1];
    const auto& split_lengths_tensor = inputs[2];
    NGRAPH_CHECK(axis_tensor->get_element_type().is_integral_number(),
                 "axis element type is not integral data type");

    int64_t axis = host_tensor_2_vector<int64_t>(axis_tensor)[0];

    axis = ngraph::normalize_axis(this, axis, data_tensor->get_partial_shape().rank());

    NGRAPH_CHECK(split_lengths_tensor->get_element_type().is_integral_number(),
                 "axis element type is not integral data type");

    std::vector<int64_t> split_lengths = host_tensor_2_vector<int64_t>(split_lengths_tensor);

    const auto data_shape = data_tensor->get_shape();
    const auto neg_one = std::find(std::begin(split_lengths), std::end(split_lengths), -1);
    if (neg_one != std::end(split_lengths)) // negative length set
    {
        const auto sum_of_known_splits =
            std::accumulate(std::begin(split_lengths), std::end(split_lengths), 0) + 1;
        split_lengths[std::distance(std::begin(split_lengths), neg_one)] =
            data_shape[axis] - sum_of_known_splits;
    }

    Shape output_shape = data_shape;
    std::vector<size_t> lower_bounds(data_shape.size(), 0);
    std::vector<size_t> upper_bounds = data_shape;
    upper_bounds.at(axis) = split_lengths[0];

    size_t split_pos = 0;
    for (const auto& output : outputs)
    {
        output_shape.at(axis) = split_lengths[split_pos++];
        output->set_shape(output_shape);
        variadic_split::evaluate(data_tensor, output, lower_bounds, upper_bounds);
        lower_bounds.at(axis) = upper_bounds.at(axis);
        if (split_pos < split_lengths.size())
            upper_bounds.at(axis) += split_lengths[split_pos];
    }

    return true;
}
bool op::v1::VariadicSplit::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_VariadicSplit_evaluate);
    return evaluate_variadic_split(inputs, outputs);
}

bool op::v1::VariadicSplit::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_VariadicSplit_has_evaluate);
    return get_input_element_type(1).is_integral_number() &&
           get_input_element_type(2).is_integral_number();
}
