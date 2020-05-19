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

#include "ngraph/op/experimental/dyn_slice.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::DynSlice::type_info;

op::DynSlice::DynSlice(const Output<Node>& arg,
                       const Output<Node>& lower_bounds,
                       const Output<Node>& upper_bounds,
                       const Output<Node>& strides,
                       const AxisSet& lower_bounds_mask,
                       const AxisSet& upper_bounds_mask,
                       const AxisSet& new_axis,
                       const AxisSet& shrink_axis,
                       const AxisSet& ellipsis_mask)
    : Op({arg, lower_bounds, upper_bounds, strides})
    , m_lower_bounds_mask(lower_bounds_mask)
    , m_upper_bounds_mask(upper_bounds_mask)
    , m_new_axis(new_axis)
    , m_shrink_axis(shrink_axis)
    , m_ellipsis_mask(ellipsis_mask)
{
    constructor_validate_and_infer_types();
}

void op::DynSlice::validate_and_infer_types()
{
    auto lower_bounds_et = get_input_element_type(1);
    auto upper_bounds_et = get_input_element_type(2);
    auto strides_et = get_input_element_type(3);

    // check data types
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
    auto lower_bounds_shape = get_input_partial_shape(1);
    auto upper_bounds_shape = get_input_partial_shape(2);
    auto strides_shape = get_input_partial_shape(3);
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

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    auto lower_bounds = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
    auto upper_bounds = as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    auto strides = as_type_ptr<op::Constant>(input_value(3).get_node_shared_ptr());

    if (lower_bounds && upper_bounds && strides)
    {
        set_output_type(0,
                        get_input_element_type(0),
                        infer_slice_shape(this,
                                          get_input_partial_shape(0),
                                          lower_bounds->cast_vector<int64_t>(),
                                          upper_bounds->cast_vector<int64_t>(),
                                          strides->cast_vector<int64_t>(),
                                          m_lower_bounds_mask,
                                          m_upper_bounds_mask,
                                          m_new_axis,
                                          m_shrink_axis,
                                          m_ellipsis_mask));
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape.rank()));
    }
}

shared_ptr<Node> op::DynSlice::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynSlice>(new_args.at(0),
                                 new_args.at(1),
                                 new_args.at(2),
                                 new_args.at(3),
                                 m_lower_bounds_mask,
                                 m_upper_bounds_mask,
                                 m_new_axis,
                                 m_shrink_axis,
                                 m_ellipsis_mask);
}

void op::DynSlice::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for DynSlice");
}
