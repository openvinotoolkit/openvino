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

#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ReplaceSlice::type_info;

op::ReplaceSlice::ReplaceSlice(const Output<Node>& arg0,
                               const Output<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Strides& strides)
    : Op({arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(strides)
{
    constructor_validate_and_infer_types();
}

op::ReplaceSlice::ReplaceSlice(const Output<Node>& arg0,
                               const Output<Node>& arg1,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds)
    : Op({arg0, arg1})
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_strides(Strides(lower_bounds.size(), 1))
{
    constructor_validate_and_infer_types();
}

void op::ReplaceSlice::validate_and_infer_types()
{
    // An empty stride vector with lower_bounds/upper_bounds filled in means that we need to
    // construct the default value.
    if (m_strides.size() == 0)
    {
        m_strides = Strides(m_lower_bounds.size(), 1);
    }

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);
    Dimension merged_args_rank;

    NODE_VALIDATION_CHECK(this,
                          Dimension::merge(merged_args_rank, arg0_shape.rank(), arg1_shape.rank()),
                          "Argument ranks do not match (arg0 shape: ",
                          arg0_shape,
                          ", arg1 shape: ",
                          arg1_shape,
                          ").");

    element::Type arg0_et = get_input_element_type(0);
    element::Type arg1_et = get_input_element_type(1);
    element::Type merged_args_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_args_et, arg0_et, arg1_et),
                          "Argument element types do not match (arg0 element type: ",
                          arg0_et,
                          ", arg1 element type: ",
                          arg1_et,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          m_lower_bounds.size() == m_upper_bounds.size() &&
                              m_lower_bounds.size() == m_strides.size(),
                          "Ranks of lower bounds (",
                          m_lower_bounds,
                          "), upper bounds (",
                          m_upper_bounds,
                          ") and strides (",
                          m_strides,
                          ") do not match.");

    size_t output_rank = m_upper_bounds.size();

    for (size_t i = 0; i < output_rank; i++)
    {
        NODE_VALIDATION_CHECK(this,
                              m_lower_bounds[i] <= m_upper_bounds[i],
                              "Lower bound for slice is greater than upper bound at axis ",
                              i,
                              " (lower bounds: ",
                              m_lower_bounds,
                              ", upper bounds: ",
                              m_upper_bounds,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              m_strides[i] != 0,
                              "Stride for slice is zero at axis ",
                              i,
                              " (strides: ",
                              m_strides,
                              ").");
    }

    NODE_VALIDATION_CHECK(this,
                          merged_args_rank.is_dynamic() ||
                              merged_args_rank.get_length() == output_rank,
                          "Argument ranks do not match the rank of the lower bounds (",
                          m_lower_bounds,
                          "), upper bounds (",
                          m_upper_bounds,
                          "), and strides (",
                          m_strides,
                          ").");

    std::vector<Dimension> sliced_dims(output_rank);

    for (size_t i = 0; i < output_rank; i++)
    {
        NODE_VALIDATION_CHECK(this,
                              arg0_shape.rank().is_dynamic() || arg0_shape[i].is_dynamic() ||
                                  m_upper_bounds[i] <= arg0_shape[i].get_length(),
                              "Upper bound for slice at axis ",
                              i,
                              " is out of range ",
                              "(upper bounds: ",
                              m_upper_bounds,
                              ", argument shape: ",
                              arg0_shape,
                              ").");

        size_t sliced_dim = m_upper_bounds[i] - m_lower_bounds[i];
        sliced_dim = sliced_dim / m_strides[i] + ((sliced_dim % m_strides[i] == 0) ? 0 : 1);
        sliced_dims[i] = sliced_dim;
    }

    PartialShape slice_shape{sliced_dims};

    NODE_VALIDATION_CHECK(this,
                          arg1_shape.compatible(slice_shape),
                          "Shape of replacement tensor (",
                          arg1_shape,
                          ") does not match the slice shape ",
                          "(",
                          slice_shape,
                          ").");

    // Slight corner case here: if arg0 was rank-unknown, we can go ahead and set the output rank
    // because the attribs will have given us enough info.
    PartialShape result_shape =
        (arg0_shape.rank().is_static())
            ? arg0_shape
            : PartialShape(std::vector<Dimension>(output_rank, Dimension::dynamic()));

    set_output_type(0, merged_args_et, result_shape);
}

shared_ptr<Node> op::ReplaceSlice::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReplaceSlice>(
        new_args.at(0), new_args.at(1), m_lower_bounds, m_upper_bounds, m_strides);
}

void op::ReplaceSlice::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);
    auto& y_element_type = y.get_element_type();
    auto y_shape = y.get_shape();

    auto zeros_shaped_like_y = op::Constant::create(y_element_type, y_shape, {0.0});

    adjoints.add_delta(x,
                       make_shared<op::ReplaceSlice>(
                           delta, zeros_shaped_like_y, m_lower_bounds, m_upper_bounds, m_strides));
    adjoints.add_delta(y, make_shared<op::Slice>(delta, m_lower_bounds, m_upper_bounds, m_strides));
}
