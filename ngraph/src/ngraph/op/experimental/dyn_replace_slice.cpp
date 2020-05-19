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

#include "ngraph/op/experimental/dyn_replace_slice.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DynReplaceSlice::type_info;

op::DynReplaceSlice::DynReplaceSlice(const Output<Node>& arg,
                                     const Output<Node>& replacement,
                                     const Output<Node>& lower_bounds,
                                     const Output<Node>& upper_bounds,
                                     const Output<Node>& strides,
                                     const AxisSet& lower_bounds_mask,
                                     const AxisSet& upper_bounds_mask,
                                     const AxisSet& new_axis,
                                     const AxisSet& shrink_axis,
                                     const AxisSet& ellipsis_mask)
    : Op({arg, replacement, lower_bounds, upper_bounds, strides})
    , m_lower_bounds_mask(lower_bounds_mask)
    , m_upper_bounds_mask(upper_bounds_mask)
    , m_new_axis(new_axis)
    , m_shrink_axis(shrink_axis)
    , m_ellipsis_mask(ellipsis_mask)
{
    constructor_validate_and_infer_types();
}

void op::DynReplaceSlice::validate_and_infer_types()
{
    auto arg_et = get_input_element_type(0);
    auto replacement_et = get_input_element_type(1);
    auto lower_bounds_et = get_input_element_type(2);
    auto upper_bounds_et = get_input_element_type(3);
    auto strides_et = get_input_element_type(4);

    element::Type result_et;

    // check data types
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, arg_et, replacement_et),
                          "Argument element type is not compatible with replacement element type");

    NODE_VALIDATION_CHECK(this,
                          lower_bounds_et.compatible(element::Type_t::i64),
                          "Lower bounds must have element type i64.");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_et.compatible(element::Type_t::i64),
                          "Upper bounds must have element type i64.");
    NODE_VALIDATION_CHECK(
        this, strides_et.compatible(element::Type_t::i64), "Strides must have element type i64");

    // check shapes
    auto arg_shape = get_input_partial_shape(0);
    auto replacement_shape = get_input_partial_shape(1);
    auto lower_bounds_shape = get_input_partial_shape(2);
    auto upper_bounds_shape = get_input_partial_shape(3);
    auto strides_shape = get_input_partial_shape(4);
    NODE_VALIDATION_CHECK(this,
                          lower_bounds_shape.rank().compatible(1),
                          "Lower bounds shape must have rank 1, got ",
                          lower_bounds_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_shape.rank().compatible(1),
                          "Upper bounds shape must have rank 1, got ",
                          upper_bounds_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          strides_shape.rank().compatible(1),
                          "Strides shape must have rank 1, got ",
                          strides_shape.rank(),
                          ".");

    PartialShape attrs_shape{PartialShape::dynamic()};

    NODE_VALIDATION_CHECK(this,
                          (lower_bounds_shape.same_scheme(PartialShape{0}) ||
                           PartialShape::merge_into(attrs_shape, lower_bounds_shape)) &&
                              (upper_bounds_shape.same_scheme(PartialShape{0}) ||
                               PartialShape::merge_into(attrs_shape, upper_bounds_shape)) &&
                              (strides_shape.same_scheme(PartialShape{0}) ||
                               PartialShape::merge_into(attrs_shape, strides_shape)),
                          "Shapes for lower bounds, upper bounds, and strides do not match");

    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);
    set_input_is_relevant_to_shape(4);

    auto lower_bounds = as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    auto upper_bounds = as_type_ptr<op::Constant>(input_value(3).get_node_shared_ptr());
    auto strides = as_type_ptr<op::Constant>(input_value(4).get_node_shared_ptr());

    // TODO(amprocte): We can get a bit more information here about the ranks of arg and
    // replacement by inspecting the attributes.
    auto slice_shape = PartialShape::dynamic();

    if (lower_bounds && upper_bounds && strides)
    {
        slice_shape = infer_slice_shape(this,
                                        get_input_partial_shape(0),
                                        lower_bounds->cast_vector<int64_t>(),
                                        upper_bounds->cast_vector<int64_t>(),
                                        strides->cast_vector<int64_t>(),
                                        m_lower_bounds_mask,
                                        m_upper_bounds_mask,
                                        m_new_axis,
                                        m_shrink_axis,
                                        m_ellipsis_mask);
    }

    NODE_VALIDATION_CHECK(this,
                          slice_shape.compatible(replacement_shape),
                          "Shape of the replacement is not compatible with the shape of the "
                          "slice (shape of slice: ",
                          slice_shape,
                          ")");

    set_output_type(0, result_et, arg_shape);
}

shared_ptr<Node> op::DynReplaceSlice::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynReplaceSlice>(new_args.at(0),
                                        new_args.at(1),
                                        new_args.at(2),
                                        new_args.at(3),
                                        new_args.at(4),
                                        m_lower_bounds_mask,
                                        m_upper_bounds_mask,
                                        m_new_axis,
                                        m_shrink_axis,
                                        m_ellipsis_mask);
}

void op::DynReplaceSlice::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                            const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for DynReplaceSlice");
}
