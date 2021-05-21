// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::ArithmeticReduction, "ArithmeticReduction", 0);

op::util::ArithmeticReduction::ArithmeticReduction() {}

op::util::ArithmeticReduction::ArithmeticReduction(const Output<Node>& arg,
                                                   const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes})
{
}

bool op::util::ArithmeticReduction::reduction_axes_constant() const
{
    return is_type<op::Constant>(input_value(1).get_node());
}

const AxisSet op::util::ArithmeticReduction::get_reduction_axes() const
{
    AxisSet axes;
    if (const auto& const_op = get_constant_from_source(input_value(1)))
    {
        const auto const_data = const_op->cast_vector<int64_t>();
        const auto input_data_rank = get_input_partial_shape(0).rank();
        const auto normalized_axes =
            ngraph::normalize_axes(get_friendly_name(), const_data, input_data_rank);
        axes = AxisSet{normalized_axes};
    }
    return axes;
}

void op::util::ArithmeticReduction::set_reduction_axes(const AxisSet& reduction_axes)
{
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector())
            ->output(0));
}

PartialShape op::util::ArithmeticReduction::infer_reduction_output_shape(const bool keep_dims)
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

void op::util::ArithmeticReduction::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(util_ArithmeticReduction_validate_and_infer_types);

    const PartialShape& axes_shape = get_input_partial_shape(1);
    const Rank axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(0) || axes_rank.compatible(1),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    PartialShape result_shape = infer_reduction_output_shape(false);
    set_input_is_relevant_to_shape(1);
    set_output_type(0, get_input_element_type(0), result_shape);
}
