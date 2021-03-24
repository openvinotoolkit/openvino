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

#include "ngraph/op/util/logical_reduction.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::util::LogicalReduction::LogicalReduction() {}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes)
    : Op({arg,
          op::Constant::create(
              element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())
              ->output(0)})
{
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg,
                                             const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes})
{
}

bool op::util::LogicalReduction::reduction_axes_constant() const
{
    return has_and_set_equal_bounds(input_value(1));
}

const AxisSet op::util::LogicalReduction::get_reduction_axes() const
{
    AxisSet axes;
    if (auto const_op = get_constant_from_source(input_value(1)))
    {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

void op::util::LogicalReduction::set_reduction_axes(const AxisSet& reduction_axes)
{
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}

void op::util::LogicalReduction::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_LogicalReduction_validate_and_infer_types);
    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    PartialShape result_shape{PartialShape::dynamic()};

    set_input_is_relevant_to_shape(1);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(element::boolean),
                          "Input element type must be boolean.");

    set_output_type(0, element::boolean, result_shape);

    if (input_rank.is_dynamic())
        return;

    if (const auto axes_const = get_constant_from_source(input_value(1)))
    {
        AxisSet reduction_axes;
        auto reduction_axes_val = axes_const->cast_vector<int64_t>();
        for (auto axis : reduction_axes_val)
        {
            try
            {
                axis = normalize_axis(this, axis, input_rank);
            }
            catch (const ngraph_error&)
            {
                NODE_VALIDATION_CHECK(this,
                                      false,
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      input_shape,
                                      ", reduction axes: ",
                                      reduction_axes,
                                      ")");
            }
            reduction_axes.insert(axis);
        }

        std::vector<Dimension> dims;
        for (size_t i = 0; i < input_rank.get_length(); i++)
        {
            if (reduction_axes.count(i) == 0)
            {
                dims.push_back(input_shape[i]);
            }
        }

        result_shape = PartialShape(dims);
    }

    set_output_type(0, element::boolean, result_shape);
}
