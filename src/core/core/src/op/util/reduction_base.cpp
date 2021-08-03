// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/reduction_base.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::util::ReductionBase, "ReductionBase", 0);

op::util::ReductionBase::ReductionBase() {}

op::util::ReductionBase::ReductionBase(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : Op({arg, reduction_axes})
{
}

PartialShape op::util::ReductionBase::infer_reduction_output_shape(const bool keep_dims)
{
    const PartialShape& data_ps = get_input_partial_shape(0);
    PartialShape result_ps{PartialShape::dynamic()};
    Rank data_rank = data_ps.rank();

    if (data_rank.is_static() && keep_dims)
    {
        result_ps = PartialShape::dynamic(data_rank);
    }

    const auto& axes = get_constant_from_source(input_value(1));
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
