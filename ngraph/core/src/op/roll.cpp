//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
                          shift_et.is_dynamic() || shift_et.is_integral_number(),
                          "Shift must have an integral number element type.");

    const auto& axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.is_dynamic() || axes_et.is_integral_number(),
                          "Axes must have an integral number element type.");

    const auto& shift_pshape = get_input_partial_shape(1);
    const auto& axes_pshape = get_input_partial_shape(2);
    const auto& shift_rank = shift_pshape.rank().get_length();
    const auto& axes_rank = axes_pshape.rank().get_length();

    NODE_VALIDATION_CHECK(this, shift_rank <= 1, "Shift must be a scalar or 1D tensor.");

    NODE_VALIDATION_CHECK(this, axes_rank <= 1, "Axes must be a scalar or 1D tensor.");

    if (!(shift_pshape.is_static() &&
          (is_scalar(shift_pshape.to_shape()) || shift_rank == 1 && shift_pshape[0] == 1)))
    {
        NODE_VALIDATION_CHECK(
            this,
            shift_pshape.compatible(axes_pshape),
            "If shift is a 1D vector, axes must be a 1D tensor of the same size.");
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
