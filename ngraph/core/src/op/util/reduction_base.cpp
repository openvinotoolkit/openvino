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
    auto data_ps = get_input_partial_shape(0);
    const auto& data_rank = data_ps.rank();
    const auto& axes = get_constant_from_source(input_value(1));
    if (data_rank.is_static() && axes)
    {
        auto axes_val = axes->cast_vector<int64_t>();
        normalize_axes(this, data_rank.get_length(), axes_val);
        if (keep_dims)
        {
            for (const auto& axis : axes_val)
                data_ps[axis] = 1;
            return data_ps;
        }
        std::vector<Dimension> dims;
        for (int64_t i = 0; i < data_rank.get_length(); ++i)
            if (find(axes_val.begin(), axes_val.end(), i) == axes_val.end())
                dims.push_back(data_ps[i]);
        return dims;
    }
    else
        return keep_dims ? PartialShape::dynamic(data_rank) : PartialShape::dynamic();
}
