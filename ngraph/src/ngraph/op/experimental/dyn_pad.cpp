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

#include "ngraph/op/experimental/dyn_pad.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DynPad::type_info;

op::DynPad::DynPad(const Output<Node>& arg,
                   const Output<Node>& padding_below,
                   const Output<Node>& padding_above,
                   const Output<Node>& padding_value,
                   op::PadMode pad_mode)
    : Op({arg, padding_below, padding_above, padding_value})
    , m_pad_mode(pad_mode)
{
    constructor_validate_and_infer_types();
}

void op::DynPad::validate_and_infer_types()
{
    auto arg_t = get_input_element_type(0);
    auto padding_value_t = get_input_element_type(3);
    NODE_VALIDATION_CHECK(
        this, arg_t.compatible(padding_value_t), "Padding value and arg type mismatch");

    // shape node should have integer data type. For now we only allow i64
    // TODO: potenially make the type more flexible to include other integer types
    auto padding_below_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          padding_below_et.compatible(element::Type_t::i64),
                          "DynPad shape must have element type i64, but has ",
                          padding_below_et);

    auto padding_above_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          padding_above_et.compatible(element::Type_t::i64),
                          "DynPad shape must have element type i64, but has ",
                          padding_above_et);

    // padding_value is of scalar shape or rank is unknown
    auto padding_value_rank = get_input_partial_shape(3).rank();
    NODE_VALIDATION_CHECK(this,
                          padding_value_rank.compatible(0),
                          "DynPad arg is not scalar (rank = 0), but has rank = ",
                          padding_value_rank);

    auto arg_shape = get_input_partial_shape(0);
    auto arg_rank = arg_shape.rank();
    auto pd_bl_shape = get_input_partial_shape(1);
    auto pd_bl_rank = pd_bl_shape.rank();
    auto pd_ab_shape = get_input_partial_shape(2);
    auto pd_ab_rank = pd_ab_shape.rank();
    auto output_rank = Rank::dynamic();

    NODE_VALIDATION_CHECK(
        this, pd_bl_rank.compatible(1), "Shape of padding below must be of rank 1");
    NODE_VALIDATION_CHECK(
        this, pd_ab_rank.compatible(1), "Shape of padding above must be of rank 1");

    if (arg_rank.is_static())
    {
        // paddings shapes should be of form {arg_rank} or dynamic
        NODE_VALIDATION_CHECK(this,
                              pd_bl_shape.compatible(PartialShape{arg_rank}),
                              "Arg and padding below ranks mismatch");

        NODE_VALIDATION_CHECK(this,
                              pd_ab_shape.compatible(PartialShape{arg_rank}),
                              "Arg and padding above ranks mismatch");
        output_rank = arg_rank;
    }
    else
    {
        // arg's rank is dynamic
        // Check padding shapes. We already know that both are either ?, {?} or {x}
        // They must be equal if both of form {x}
        NODE_VALIDATION_CHECK(
            this, pd_bl_shape.compatible(pd_ab_shape), "Padding below and above ranks mismatch");

        output_rank = pd_bl_shape.is_static() ? pd_bl_shape[0] : pd_ab_shape[0];
    }

    auto out_shape = PartialShape::dynamic(output_rank);

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_output_type(0, arg_t, out_shape);
}

shared_ptr<Node> op::DynPad::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynPad>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_pad_mode);
}

// TODO: This function is not implemented!
void op::DynPad::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                   const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for DynPad");
}
