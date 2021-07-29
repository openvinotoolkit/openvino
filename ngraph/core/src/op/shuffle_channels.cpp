// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/shuffle_channels.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/shuffle_channels.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v0::ShuffleChannels, "ShuffleChannels", 0);

op::ShuffleChannels::ShuffleChannels(const Output<Node>& data,
                                     const int64_t axis,
                                     const int64_t group)
    : Op({data})
    , m_axis(axis)
    , m_group{group}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ShuffleChannels::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("group", m_group);
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

void op::ShuffleChannels::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
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

        NODE_VALIDATION_CHECK(
            this, m_group >= 1, "The 'group' parameter must be greater or equal to 1.");

        const auto channel_dim_size = shape.at(axis_zb);
        NODE_VALIDATION_CHECK(
            this,
            channel_dim_size % m_group == 0,
            "The channel dimension size has to be a multiple of the groups parameter value.");
    }
    set_output_type(0, data_type, get_input_partial_shape(0));
}

shared_ptr<Node> op::ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    if (new_args.size() != 1)
    {
        throw ngraph_error("Expected 1 element in new_args for the ShuffleChannels op but got " +
                           std::to_string(new_args.size()));
    }

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

bool op::ShuffleChannels::evaluate_shuffle_channels(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const
{
    const auto arg = inputs[0]->get_data_ptr<const char>();
    auto out = outputs[0]->get_data_ptr<char>();
    const auto data_shape = inputs[0]->get_shape();
    const size_t elem_size = inputs[0]->get_element_type().size();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(data_shape);

    runtime::reference::shuffle_channels(arg, out, data_shape, elem_size, m_axis, m_group);

    return true;
}
bool op::ShuffleChannels::evaluate(const HostTensorVector& outputs,
                                   const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_evaluate);
    return evaluate_shuffle_channels(outputs, inputs);
}

bool op::ShuffleChannels::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v0_ShuffleChannels_has_evaluate);
    return true;
}
