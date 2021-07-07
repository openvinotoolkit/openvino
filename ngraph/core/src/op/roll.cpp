// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/roll.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v7::Roll, "Roll", 7);

op::v7::Roll::Roll(const Output<Node>& data, const Output<Node>& shift, const Output<Node>& axes)
    : Op({data, shift, axes})
{
    constructor_validate_and_infer_types();
}

void op::v7::Roll::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v7_Roll_validate_and_infer_types);

    const auto& shift_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shift_et.is_dynamic() || shift_et == element::i32 ||
                              shift_et == element::i64,
                          "Shift must have int32 or int64 element type.");

    const auto& axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.is_dynamic() || axes_et == element::i32 ||
                              axes_et == element::i64,
                          "Axes must have int32 or int64 element type.");

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& shift_pshape = get_input_partial_shape(1);
    const auto& axes_pshape = get_input_partial_shape(2);

    if (shift_pshape.is_static())
    {
        const auto& shift_rank = shift_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this, shift_rank <= 1, "Shift must be a scalar or 1D tensor.");
    }

    if (axes_pshape.is_static())
    {
        const auto& axes_rank = axes_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this, axes_rank <= 1, "Axes must be a scalar or 1D tensor.");
    }

    // If shift is a scalar, than axes can be arbitrary 1d tensor and we don't need
    // to check shift shape consistency with axes, otherwise the check is needed.
    if (!(shift_pshape.is_static() && is_scalar(shift_pshape.to_shape())))
    {
        NODE_VALIDATION_CHECK(
            this,
            shift_pshape.compatible(axes_pshape),
            "If shift is a 1D vector, axes must be a 1D tensor of the same size.");
    }

    if (const auto& const_axes = get_constant_from_source(input_value(2)))
    {
        auto axes = const_axes->cast_vector<int64_t>();

        if (data_pshape.is_static())
        {
            const auto& data_rank = data_pshape.rank().get_length();
            for (int64_t& axis : axes)
            {
                NODE_VALIDATION_CHECK(this,
                                      axis < data_rank,
                                      "Axes must be less than data tensor rank. Got "
                                      "data tensor rank: ",
                                      data_rank,
                                      ", axis: ",
                                      axis);
                if (axis < 0)
                {
                    axis += data_rank;
                }
                NODE_VALIDATION_CHECK(this,
                                      axis >= 0,
                                      "Axes must be positive or equal to zero. Got "
                                      "axis: ",
                                      axis);
            }
        }
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

bool op::v7::Roll::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v7_Roll_visit_attributes);
    return true;
}

shared_ptr<Node> op::v7::Roll::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v7_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Roll>(new_args[0], new_args[1], new_args[2]);
}
