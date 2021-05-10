// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/arithmetic_reductions_keep_dims.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::ArithmeticReductionKeepDims, "ArithmeticReductionKeepDims", 0);

op::util::ArithmeticReductionKeepDims::ArithmeticReductionKeepDims(
    const ngraph::Output<ngraph::Node>& arg,
    const ngraph::Output<ngraph::Node>& reduction_axes,
    bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{keep_dims}
{
}

bool ngraph::op::util::ArithmeticReductionKeepDims::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_util_ArithmeticReductionKeepDims_visit_attributes);
    visitor.on_attribute("keep_dims", m_keep_dims);
    return true;
}

void op::util::ArithmeticReductionKeepDims::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_util_ArithmeticReductionKeepDims_validate_and_infer_types);

    const element::Type& data_et = get_input_element_type(0);

    const PartialShape& axes_shape = get_input_partial_shape(1);
    const element::Type& axes_et = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          data_et.is_real() || data_et.is_integral_number(),
                          "Element type of data input must be numeric. Got: ",
                          data_et);

    NODE_VALIDATION_CHECK(this,
                          axes_et.compatible(element::i64) || axes_et.compatible(element::i32),
                          "Element type of axes input must be either i64 or i32. Got: ",
                          axes_et);

    NODE_VALIDATION_CHECK(this,
                          axes_shape.compatible(PartialShape{}) ||
                              axes_shape.compatible(PartialShape{0}) ||
                              axes_shape.compatible(PartialShape::dynamic(1)),
                          "Axes input must be a scalar or 1D input. Got: ",
                          axes_shape);

    if (m_keep_dims)
    {
        auto input_shape = get_input_partial_shape(0);
        auto input_rank = input_shape.rank();
        PartialShape result_shape{PartialShape::dynamic()};

        if (input_rank.is_static())
            result_shape = PartialShape::dynamic(input_rank);

        const auto& axes = get_constant_from_source(input_value(1));
        if (input_rank.is_static() && axes)
        {
            AxisSet reduction_axes;
            auto reduction_axes_val = axes->cast_vector<int64_t>();

            bool unique_axes_val =
                std::set<int64_t>(reduction_axes_val.begin(), reduction_axes_val.end()).size() ==
                reduction_axes_val.size();

            NODE_VALIDATION_CHECK(
                this, unique_axes_val, "Axes input must have unique axis values.");

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
            for (int64_t i = 0; i < input_rank.get_length(); i++)
            {
                if (reduction_axes.count(i) == 0)
                {
                    dims.push_back(input_shape[i]);
                }
                else
                {
                    dims.emplace_back(Dimension{1});
                }
            }
            result_shape = PartialShape(dims);
        }
        set_input_is_relevant_to_shape(1);
        set_output_type(0, get_input_element_type(0), result_shape);
    }
    else
    {
        ArithmeticReduction::validate_and_infer_types();
    }
}
