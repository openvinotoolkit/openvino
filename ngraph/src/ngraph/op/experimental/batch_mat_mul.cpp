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

#include "batch_mat_mul.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::BatchMatMul::type_info;

op::v0::BatchMatMul::BatchMatMul(const Output<Node>& arg0, const Output<Node>& arg1)
    : Op({arg0, arg1})
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::BatchMatMul::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<BatchMatMul>(new_args.at(0), new_args.at(1));
}

void op::v0::BatchMatMul::validate_and_infer_types()
{
    // Check input types
    const auto& arg0_et = get_input_element_type(0);
    const auto& arg1_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          arg0_et.compatible(arg1_et),
                          "Inputs arg0 and arg1 must have compatible element type.");
    // Check input shapes
    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          arg0_shape.rank().compatible(3),
                          "Input arg0 shape must have rank 3, got ",
                          arg0_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          arg1_shape.rank().compatible(3),
                          "Input arg1 shape must have rank 3, got ",
                          arg1_shape.rank(),
                          ".");

    // We expect output shape always have rank 3
    PartialShape output_shape(PartialShape::dynamic(3));

    // Construct output shape with more information if avalible.
    if (arg0_shape.rank().same_scheme(3) && arg1_shape.rank().same_scheme(3))
    {
        NODE_VALIDATION_CHECK(this,
                              arg0_shape[0].compatible(arg1_shape[0]),
                              "Batch size dimensions are not equal while creating BatchMatMul.");
        NODE_VALIDATION_CHECK(this,
                              arg0_shape[2].compatible(arg1_shape[1]),
                              "Product dimensions are not equal while creating BatchMatMul.");
        auto batch_size = arg0_shape[0].is_static() ? arg0_shape[0] : arg1_shape[0];
        output_shape = PartialShape{batch_size, arg0_shape[1], arg1_shape[2]};
    }
    auto output_et = arg0_et.is_dynamic() ? arg1_et : arg0_et;
    set_output_type(0, output_et, output_shape);
}

void op::v0::BatchMatMul::generate_adjoints(autodiff::Adjoints& adjoints,
                                            const OutputVector& deltas)
{
    auto delta = deltas.at(0); // NxIxK

    auto arg0 = input_value(0); // NxIxJ
    auto arg1 = input_value(1); // NxJxK

    auto delta_dot_arg1 = make_shared<op::BatchMatMul>(
        delta, util::batch_mat_transpose(arg1.get_node_shared_ptr())); // IK.KJ->IJ
    adjoints.add_delta(arg0, delta_dot_arg1);

    auto arg0_dot_delta = make_shared<BatchMatMul>(
        util::batch_mat_transpose(arg0.get_node_shared_ptr()), delta); // JI.IK->JK
    adjoints.add_delta(arg1, arg0_dot_delta);
}

shared_ptr<Node> op::util::batch_mat_transpose(const shared_ptr<Node>& node)
{
    const auto& node_shape = node->get_output_partial_shape(0);
    // index 0 is the batch, only transposing the others.
    if (node_shape.is_static())
    {
        // Applies static shape transpose
        Shape static_shape = node_shape.to_shape();
        std::swap(static_shape[1], static_shape[2]);
        return make_shared<op::Reshape>(node, AxisVector{0, 2, 1}, static_shape);
    }
    else
    {
        // Applies dynamic transpose
        // XXX lfeng: to be implemented using reshape that supports PartialShape
        throw ngraph_error(
            "generate_adjoints not implemented for BatchMatMulTranspose with dynamic input shapes");
    }
}
