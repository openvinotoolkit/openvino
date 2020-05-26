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

#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DynBroadcast::type_info;

op::DynBroadcast::DynBroadcast(const Output<Node>& arg,
                               const Output<Node>& shape,
                               const Output<Node>& broadcast_axes)
    : Op({arg, shape, broadcast_axes})
{
    constructor_validate_and_infer_types();
}

void op::DynBroadcast::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    // TODO: potenially make the type more flexible to include other integer types
    auto shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.compatible(element::Type_t::i64),
                          "DynBroadcast shape must have element type i64, but has ",
                          shape_et);

    // shape node should produce a one dimensional shape.
    auto broadcast_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_rank.compatible(1),
                          "DynBroadcast shape rank must be 1, but has ",
                          broadcast_shape_rank);

    // axes node should have integer data type. For now we only allow i64
    // TODO: potenially make the type more flexible to include other integer types
    auto axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.compatible(element::Type_t::i64),
                          "DynBroadcast axes must have element type i64, but has ",
                          axes_et);

    // axes node should produce a one dimensional shape.
    auto axes_shape_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          axes_shape_rank.compatible(1),
                          "DynBroadcast axes rank must be 1, but has ",
                          axes_shape_rank);

    PartialShape result_shape{PartialShape::dynamic()};
    if (is_type<op::v0::Constant>(input_value(1).get_node_shared_ptr()))
    {
        result_shape = static_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())
                           ->get_shape_val();
    }

    bool axes_known = false;
    AxisSet broadcast_axes;
    if (is_type<op::v0::Constant>(input_value(2).get_node_shared_ptr()))
    {
        axes_known = true;
        broadcast_axes = static_pointer_cast<op::Constant>(input_value(2).get_node_shared_ptr())
                             ->get_axis_set_val();
    }

    PartialShape arg_shape = get_input_partial_shape(0);
    if (result_shape.is_static() && axes_known && arg_shape.is_static())
    {
        for (auto axis : broadcast_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < result_shape.rank().get_length(),
                                  "Broadcast axis index (",
                                  axis,
                                  ") exceeds specified output shape rank ",
                                  "(broadcast axes: ",
                                  broadcast_axes,
                                  ", output shape: ",
                                  result_shape,
                                  ").");
        }

        Shape required_input_shape = result_shape.to_shape();
        for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); ++i)
        {
            required_input_shape.erase(required_input_shape.begin() + *i);
        }

        // TODO(amprocte): We can probably have a more helpful error message here.
        // There are two things that can go wrong, which are being picked up in
        // one fell swoop by this check: either the number of broadcast axes is not
        // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
        NODE_VALIDATION_CHECK(
            this,
            arg_shape.compatible(required_input_shape),
            "Broadcast argument shape, specified output shape, and axes are incompatible ",
            "(argument shape: ",
            arg_shape,
            ", output shape: ",
            result_shape,
            ", broadcast axes: ",
            broadcast_axes,
            ").");
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::DynBroadcast::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynBroadcast>(new_args.at(0), new_args.at(1), new_args.at(2));
}

// TODO: This function is not implemented!
void op::DynBroadcast::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                         const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for DynBroadcast");
}
