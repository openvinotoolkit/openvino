// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/logical_reduction.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::LogicalReduction, "LogicalReduction", 1);

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

PartialShape op::util::LogicalReduction::infer_reduction_output_shape(const bool keep_dims)
{
    const PartialShape& data_ps = get_input_partial_shape(0);
    PartialShape result_ps{PartialShape::dynamic()};
    Rank data_rank = data_ps.rank();

    if (data_rank.is_static() && keep_dims)
    {
        result_ps = PartialShape::dynamic(data_rank);
    }

    const auto& axes = get_constant_from_source(this->input_value(1));
    if (data_rank.is_static() && axes)
    {
        AxisSet reduction_axes;
        auto reduction_axes_val = axes->cast_vector<int64_t>();
        for (auto axis : reduction_axes_val)
        {
            try
            {
                axis = normalize_axis(this, axis, data_rank);
            }
            catch (const ngraph_error&)
            {
                NODE_VALIDATION_CHECK(this,
                                      false,
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds ",
                                      "(argument shape: ",
                                      data_ps,
                                      ", reduction axes: ",
                                      reduction_axes,
                                      ")");
            }
            reduction_axes.insert(axis);
        }
        std::vector<Dimension> dims;
        for (int64_t i = 0; i < data_rank.get_length(); i++)
        {
            if (reduction_axes.count(i) == 0)
            {
                dims.push_back(data_ps[i]);
            }
            else if (keep_dims)
            {
                dims.emplace_back(Dimension{1});
            }
        }
        result_ps = PartialShape(dims);
    }
    return result_ps;
}

void op::util::LogicalReduction::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_LogicalReduction_validate_and_infer_types);

    const element::Type& data_et = get_input_element_type(0);
    const PartialShape& axes_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(
        this, data_et.compatible(element::boolean), "Element type of data input must be boolean.");

    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(false);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, data_et, result_shape);
}
