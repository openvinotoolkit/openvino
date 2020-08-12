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

#include <numeric>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/validation_util.hpp"

#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/slice.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::VariadicSplit::type_info;

op::v1::VariadicSplit::VariadicSplit(const Output<Node>& data,
                                     const Output<Node>& axis,
                                     const Output<Node>& split_lengths)
    : Op({data, axis, split_lengths})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::VariadicSplit::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void ngraph::op::v1::VariadicSplit::validate_and_infer_types()
{
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
        auto axis_input = input_value(1).get_node_shared_ptr();
        auto split_lengths_input = input_value(2).get_node_shared_ptr();
        auto data_shape = data.get_partial_shape();
        const auto& data_type = data.get_element_type();

        set_output_size(num_outputs);
        if (data_shape.rank().is_static() && op::is_constant(axis_input) &&
            op::is_constant(split_lengths_input))
        {
            const auto axis_input_constant = as_type_ptr<op::Constant>(axis_input);
            auto axis_val = axis_input_constant->cast_vector<int64_t>()[0];

            // Adjust split axis in case of negatives
            int64_t axis = ngraph::normalize_axis(this, axis_val, data_shape.rank());

            auto split_lengths =
                as_type_ptr<op::Constant>(split_lengths_input)->cast_vector<int64_t>();
            // Adjust split lengths in case of negatives
            size_t sum_of_splits = 0;
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

            for (size_t output{0}; output < num_outputs; ++output)
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
            for (size_t output{0}; output < num_outputs; ++output)
            {
                set_output_type(output, data_type, PartialShape::dynamic());
            }
        }
    }
}

shared_ptr<Node> op::v1::VariadicSplit::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::VariadicSplit>(new_args.at(0), new_args.at(1), new_args.at(2));
}

namespace
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

    bool evaluate_variadic_split(const HostTensorPtr& data_tensor,
                                 const HostTensorPtr& axis_tensor,
                                 const HostTensorPtr& split_lengths_tensor,
                                 const HostTensorVector& outputs,
                                 const Node* split_node)
    {
        int64_t axis;
        switch (axis_tensor->get_element_type())
        {
        case element::Type_t::i16: axis = read_vector<int16_t>(axis_tensor)[0]; break;
        case element::Type_t::i32: axis = read_vector<int32_t>(axis_tensor)[0]; break;
        case element::Type_t::i64: axis = read_vector<int64_t>(axis_tensor)[0]; break;
        case element::Type_t::u64:
            axis = static_cast<int64_t>(read_vector<uint64_t>(axis_tensor)[0]);
            break;
        default:
            NODE_VALIDATION_CHECK(split_node,
                                  false,
                                  "Not supported axis type: ",
                                  axis_tensor->get_element_type(),
                                  " during evaluate Split:v1");
            break;
        }
        axis = ngraph::normalize_axis(split_node, axis, data_tensor->get_partial_shape().rank());

        std::vector<int64_t> split_lengths;
        switch (split_lengths_tensor->get_element_type())
        {
        case element::Type_t::i32:
        {
            const auto split_lengths_i32 = read_vector<int32_t>(split_lengths_tensor);
            split_lengths =
                std::vector<int64_t>(std::begin(split_lengths_i32), std::end(split_lengths_i32));
            break;
        }
        case element::Type_t::i64:
        {
            const auto split_lengths_i64 = read_vector<int64_t>(split_lengths_tensor);
            split_lengths =
                std::vector<int64_t>(std::begin(split_lengths_i64), std::end(split_lengths_i64));
            break;
        }
        case element::Type_t::u64:
        {
            const auto split_lengths_u64 = read_vector<uint64_t>(split_lengths_tensor);
            split_lengths =
                std::vector<int64_t>(std::begin(split_lengths_u64), std::end(split_lengths_u64));
            break;
        }
        default:
            NODE_VALIDATION_CHECK(split_node,
                                  false,
                                  "Not supported split lengths type: ",
                                  split_lengths_tensor->get_element_type(),
                                  " during evaluate Split:v1");
            break;
        }

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

        int64_t split_pos = 0;
        for (const auto& output : outputs)
        {
            output_shape.at(axis) = split_lengths[split_pos++];
            output->set_shape(output_shape);
            evaluate(data_tensor, output, lower_bounds, upper_bounds);
            lower_bounds.at(axis) = upper_bounds.at(axis);
            upper_bounds.at(axis) += split_lengths[split_pos];
        }

        return true;
    }
}

bool op::v1::VariadicSplit::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    const auto& data = inputs[0];
    const auto& axis = inputs[1];
    const auto& split_lengths = inputs[2];

    return evaluate_variadic_split(data, axis, split_lengths, outputs, this);
}
