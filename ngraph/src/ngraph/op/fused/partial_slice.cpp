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

#include <cmath>
#include <numeric>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/partial_slice.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PartialSlice::type_info;

op::PartialSlice::PartialSlice(const Output<Node>& data,
                               const AxisVector& axes,
                               const std::vector<int64_t>& lower_bounds,
                               const std::vector<int64_t>& upper_bounds,
                               const AxisVector& decrease_axes)
    : FusedOp({data})
    , m_axes(axes)
    , m_lower_bounds(lower_bounds)
    , m_upper_bounds(upper_bounds)
    , m_decrease_axes(decrease_axes)
{
    constructor_validate_and_infer_types();
}

// All input shape should be static by this point
OutputVector op::PartialSlice::decompose_op() const
{
    const PartialShape& data_pshape = get_input_partial_shape(0);
    if (data_pshape.is_dynamic())
    {
        throw ngraph_error("Data needs to have static shape to decompose");
    }

    auto data = input_value(0);
    auto data_shape = data.get_shape();
    auto axes = get_axes();
    auto starts = get_lower_bounds();
    auto ends = get_upper_bounds();
    auto decrease_axes = get_decrease_axes();

    Coordinate ng_start, ng_end;
    int axis_length, start, end;
    for (size_t i = 0; i < data_shape.size(); ++i)
    {
        ng_start.push_back(0);
        ng_end.push_back(data_shape[i]);
    }

    for (size_t i = 0; i < axes.size(); ++i)
    {
        axis_length = data_shape[axes[i]];
        start = starts[i] < 0 ? (starts[i] + axis_length) : starts[i];
        end = ends[i] < 0 ? (ends[i] + axis_length) : ends[i];
        start = max(start, 0);
        end = max(end, 0);
        start = min(start, axis_length);
        end = min(end, axis_length);
        start = min(start, end);
        ng_start[axes[i]] = start;
        ng_end[axes[i]] = end;
    }

    auto sliced = std::make_shared<op::Slice>(data, ng_start, ng_end);
    auto out_shape = sliced->get_shape();
    Shape out_reshape_shape{};

    if (decrease_axes.size() > 0)
    {
        auto new_out_shape = out_shape;
        for (size_t i = 0; i < decrease_axes.size(); ++i)
        {
            int idx = decrease_axes[i];
            NGRAPH_CHECK(out_shape[idx] == 1, "Decrease dim should be 1");
            new_out_shape[idx] = 0;
        }

        for (size_t i = 0; i < out_shape.size(); ++i)
        {
            if (new_out_shape[i] != 0)
            {
                out_reshape_shape.push_back(out_shape[i]);
            }
        }

        if (out_reshape_shape.size() == 0)
        {
            out_reshape_shape.push_back(1);
        }
    }
    else
    {
        out_reshape_shape = out_shape;
    }

    auto out =
        std::make_shared<op::Reshape>(sliced, get_default_order(out_shape), out_reshape_shape);
    return {out};
}

shared_ptr<Node> op::PartialSlice::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<PartialSlice>(
        new_args.at(0), m_axes, m_lower_bounds, m_upper_bounds, m_decrease_axes);
}

void op::PartialSlice::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape data_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (data_pshape.is_dynamic())
    {
        set_output_type(0, input_element_type, PartialShape::dynamic());
    }
}
