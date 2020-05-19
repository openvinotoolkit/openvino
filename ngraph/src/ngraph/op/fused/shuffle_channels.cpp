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

#include "ngraph/op/fused/shuffle_channels.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::ShuffleChannels::type_info;

op::ShuffleChannels::ShuffleChannels(const Output<Node>& data, const int axis, const size_t groups)
    : FusedOp({data})
    , m_axis(axis)
    , m_groups{groups}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ShuffleChannels::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("groups", m_groups);
    return true;
}

size_t op::ShuffleChannels::get_zero_based_axis() const
{
    if (m_axis >= 0)
    {
        return m_axis;
    }
    else
    {
        if (!get_input_partial_shape(0).rank().is_dynamic())
        {
            return m_axis + get_input_partial_shape(0).rank().get_length();
        }
        else
        {
            throw ngraph_error("Cannot request zero-based axis with a input of unknown rank");
        }
    }
}

void op::ShuffleChannels::pre_validate_and_infer_types()
{
    if (get_input_partial_shape(0).is_static())
    {
        const auto shape = get_input_shape(0);

        NODE_VALIDATION_CHECK(
            this, shape.size() >= 1, "The input tensor's shape is expected to be at least 1D.");
        size_t axis_zb = get_zero_based_axis();

        NODE_VALIDATION_CHECK(this,
                              axis_zb < shape.size(),
                              "The 'axis' parameter for ShuffleChannels has to point to one of the "
                              "input tensor's shape dimensions.");

        const auto channel_dim_size = shape.at(axis_zb);
        NODE_VALIDATION_CHECK(
            this,
            channel_dim_size % m_groups == 0,
            "The channel dimension size has to be a multiple of the groups parameter value.");
    }
}

NodeVector op::ShuffleChannels::decompose_op() const
{
    const auto data = input_value(0);
    const auto& data_shape = data.get_shape();

    const auto reshaped = builder::reshape(data, get_pre_shuffle_shape(data_shape));
    const auto shuffled = builder::reorder_axes(reshaped, {0, 2, 1, 3});

    return {builder::reshape(shuffled, data_shape)};
}

shared_ptr<Node> op::ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Expected 1 element in new_args for the ShuffleChannels op but got " +
                           std::to_string(new_args.size()));
    }

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_groups);
}

Shape op::ShuffleChannels::get_pre_shuffle_shape(const Shape& data_shape) const
{
    const Shape& ds = data_shape;

    // in general the resulting shape should contain the following values:
    // [0]: ds[0] * ds[1] * ... * ds[m_axis-1] (or 1 if m_axis == 0)
    // [1]: m_groups
    // [2]: ds[axis] / m_groups
    // [3]: ds[axis+1] * ds[axis+2] * ... * ds[ds.size()-1] (or 1 if m_axis points to the last elem
    //                                                       of ds)
    Shape res(4, 1);

    size_t axis_zb = get_zero_based_axis();
    for (size_t i = 0; i < axis_zb; ++i)
    {
        res[0] *= ds[i];
    }

    res[1] = m_groups;
    res[2] = ds[axis_zb] / m_groups;

    for (size_t i = axis_zb + 1; i < ds.size(); ++i)
    {
        res[3] *= ds[i];
    }

    return res;
}
