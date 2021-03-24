// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/idft.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v7::IDFT, "IDFT", 7);

op::v7::IDFT::IDFT(const Output<Node>& data, const Output<Node>& axes)
    : Op({data, axes})
{
    constructor_validate_and_infer_types();
}

op::v7::IDFT::IDFT(const Output<Node>& data,
                   const Output<Node>& axes,
                   const Output<Node>& signal_size)
    : Op({data, axes, signal_size})
{
    constructor_validate_and_infer_types();
}

bool op::v7::IDFT::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_IDFT_visit_attributes);
    return true;
}

std::shared_ptr<Node> op::v7::IDFT::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_IDFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(
        this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2)
    {
        return std::make_shared<op::v7::IDFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<op::v7::IDFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::v7::IDFT::validate()
{
    size_t num_of_inputs = get_input_size();

    NODE_VALIDATION_CHECK(
        this, num_of_inputs == 2 || num_of_inputs == 3, "IDFT must have 2 or 3 inputs.");

    element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 ||
                              input_et == element::bf16,
                          "IDFT input element type must be f32, f16, or bf16");

    element::Type axes_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_et == element::i64 || axes_et == element::i32,
                          "IDFT axes element type must be i32 or i64");

    const auto& input_shape = PartialShape(get_input_partial_shape(0));
    if (input_shape.rank().is_static())
    {
        const auto input_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(this,
                              input_rank >= 2,
                              "The input rank must be greater or equal to 2. Got input rank: ",
                              input_rank);

        auto last_dim_with_two = input_shape[input_rank - 1] & Dimension(2);
        NODE_VALIDATION_CHECK(this,
                              !last_dim_with_two.get_interval().empty(),
                              "The last dimension of input data must be 2. Got: ",
                              input_shape[input_rank - 1]);
    }

    const auto& axes_shape = PartialShape(get_input_partial_shape(1));
    if (axes_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              axes_shape.rank().get_length() == 1,
                              "IDFT axes input must be 1D tensor. Got axes input rank: ",
                              axes_shape.rank().get_length());
    }

    if (input_shape.rank().is_static() && axes_shape.is_static())
    {
        const auto input_rank = input_shape.rank().get_length();
        NODE_VALIDATION_CHECK(this,
                              input_rank >= axes_shape.to_shape()[0] + 1,
                              "The input rank must be greater than number of IDFT axes. Got "
                              "input rank: ",
                              input_rank,
                              ", number of axes: ",
                              axes_shape.to_shape()[0]);
    }

    if (input_shape.rank().is_static() && is_type<op::Constant>(input_value(1).get_node()))
    {
        const auto input_rank = input_shape.rank().get_length();
        const auto& const_axes = get_constant_from_source(input_value(1));
        auto axes = const_axes->cast_vector<int64_t>();

        // IDFT operation supports for negative axes to transform. More precisely, according to
        // the IDFT operation specification, axes should be integers from -(r - 1) to (r - 2)
        // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
        //'r - 1 + a'. The reason is the following.
        for (int64_t& axis : axes)
        {
            if (axis < 0)
            {
                axis += input_rank - 1;
            }
        }

        AxisVector axes_vector;
        AxisSet axes_set;
        for (const int64_t axis : axes)
        {
            axes_vector.push_back(static_cast<size_t>(axis));
            axes_set.insert(static_cast<size_t>(axis));
        }

        NODE_VALIDATION_CHECK(
            this, axes.size() == axes_set.size(), "IDFT axes must be unique. Got: ", axes_vector);

        NODE_VALIDATION_CHECK(this,
                              std::find(axes.begin(), axes.end(), input_rank - 1) == axes.end(),
                              "IDFT axes cannot contain the last axis. Got axes: ",
                              axes_vector);
    }

    if (num_of_inputs == 3)
    {
        element::Type signal_size_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              signal_size_et == element::i64 || signal_size_et == element::i32,
                              "IDFT signal_size element type must be i32 or i64");

        const auto& signal_size_shape = PartialShape(get_input_partial_shape(2));
        if (signal_size_shape.rank().is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  signal_size_shape.rank().get_length() == 1,
                                  "IDFT Signal size input must be 1D tensor. Got signal size "
                                  "input rank: ",
                                  signal_size_shape.rank().get_length());
        }

        if (axes_shape.is_static() && signal_size_shape.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  axes_shape.to_shape()[0] == signal_size_shape.to_shape()[0],
                                  "Sizes of inputs 'axes' and 'signal_size' must be equal. Got "
                                  "size of 'axes': ",
                                  axes_shape.to_shape()[0],
                                  "size of 'signal_size': ",
                                  signal_size_shape.to_shape()[0]);
        }
    }
}

void op::v7::IDFT::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_IDFT_validate_and_infer_types);
    validate();

    const auto& input_shape = PartialShape(get_input_partial_shape(0));
    const auto& axes_shape = PartialShape(get_input_partial_shape(1));
    PartialShape output_shape = input_shape;
    if (input_shape.rank().is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    const auto input_rank = input_shape.rank().get_length();

    if (axes_shape.rank().is_dynamic() || !is_type<op::Constant>(input_value(1).get_node()))
    {
        for (size_t i = 0; i < input_rank - 1; ++i)
        {
            output_shape[i] = Dimension::dynamic();
        }
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    if (input_values().size() == 2)
    {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    const auto& signal_size_shape = PartialShape(get_input_partial_shape(2));
    if (signal_size_shape.rank().is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    const auto& const_axes = get_constant_from_source(input_value(1));
    auto axes = const_axes->cast_vector<int64_t>();
    // IDFT operation supports for negative axes to transform. More precisely, according to
    // the IDFT operation specification, axes should be integers from -(r - 1) to (r - 2)
    // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
    //'r - 1 + a'. The reason is the following.
    for (int64_t& axis : axes)
    {
        if (axis < 0)
        {
            axis += input_rank - 1;
        }
    }

    if (!is_type<op::Constant>(input_value(2).get_node()))
    {
        for (int64_t axis : axes)
        {
            output_shape[axis] = Dimension::dynamic();
        }
        set_output_type(0, get_input_element_type(0), output_shape);
        return;
    }

    const auto& const_signal_size = get_constant_from_source(input_value(2));
    const auto signal_size = const_signal_size->cast_vector<int64_t>();

    size_t num_of_axes = axes.size();
    for (size_t i = 0; i < num_of_axes; ++i)
    {
        if (signal_size[i] == -1)
        {
            continue;
        }
        output_shape[axes[i]] = Dimension(signal_size[i]);
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}
