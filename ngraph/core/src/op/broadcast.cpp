// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/partial_shape.hpp"

#include <ngraph/validation_util.hpp>
#include <numeric>
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::Broadcast::type_info;

op::v3::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const Output<Node>& axes_mapping,
                             const BroadcastModeSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, axes_mapping, broadcast_spec}
{
    constructor_validate_and_infer_types();
}

op::v3::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const BroadcastModeSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, broadcast_spec}
{
    constructor_validate_and_infer_types();
}

namespace
{
    std::pair<bool, AxisSet> get_broadcast_axes_bidirectional(const Shape& arg_shape,
                                                              const Shape& result_shape)
    {
        AxisSet broadcast_axes;
        bool axes_known = false;
        const auto start_axis = result_shape.size() - arg_shape.size();
        NGRAPH_CHECK(start_axis >= 0);
        for (size_t i = 0; i < result_shape.size(); i++)
        {
            if (i < start_axis || result_shape[i] != arg_shape[i - start_axis])
            {
                broadcast_axes.insert(i);
            }
        }
        axes_known = true;
        return std::make_pair(axes_known, broadcast_axes);
    }
} // namespace

std::pair<bool, AxisSet> op::v3::Broadcast::get_broadcast_axes() const
{
    if (m_mode.m_type == BroadcastType::BIDIRECTIONAL)
    {
        AxisSet broadcast_axes;
        bool axes_known = false;

        if (get_input_partial_shape(0).is_static() && get_output_partial_shape(0).is_static())
        {
            const auto arg_shape = get_input_shape(0);
            const auto result_shape = get_output_shape(0);
            return get_broadcast_axes_bidirectional(arg_shape, result_shape);
        }
        return std::make_pair(axes_known, broadcast_axes);
    }

    return util::BroadcastBase::get_broadcast_axes();
}

namespace
{
    PartialShape get_result_shape_bidirectional(const Node* this_ptr,
                                                const PartialShape& arg_shape,
                                                Shape& target_shape)
    {
        if (arg_shape.rank().is_dynamic())
        {
            return PartialShape::dynamic();
        }
        auto arg_shape_vec = static_cast<std::vector<Dimension>>(arg_shape);
        PartialShape result_shape;
        // Add left padding to shorter target or argument shape
        const auto target_padded_rank = std::max(arg_shape_vec.size(), target_shape.size());
        while (arg_shape_vec.size() < target_padded_rank)
        {
            arg_shape_vec.insert(arg_shape_vec.begin(), 1);
        }
        while (target_shape.size() < target_padded_rank)
        {
            target_shape.insert(target_shape.begin(), 1);
        }

        result_shape = target_shape;
        for (size_t i = 0; i < target_shape.size(); ++i)
        {
            if (arg_shape_vec[i].is_dynamic())
            {
                if (target_shape[i] == 1)
                {
                    result_shape[i] = Dimension::dynamic();
                }
                else
                {
                    result_shape[i] = target_shape[i];
                }
                continue;
            }
            const size_t arg_shape_dim = arg_shape_vec[i].get_length();
            NODE_VALIDATION_CHECK(this_ptr,
                                  arg_shape_dim == 1 || target_shape[i] == 1 ||
                                      arg_shape_dim == target_shape[i],
                                  "Broadcast incorrect target shape. Expecting either 1 or ",
                                  arg_shape_dim,
                                  ". Got ",
                                  target_shape[i]);

            result_shape[i] = std::max(arg_shape_dim, target_shape[i]);
        }
        return result_shape;
    }
} // namespace

bool op::v3::Broadcast::broadcast_evaluate(const HostTensorVector& outputs,
                                           const HostTensorVector& inputs) const
{
    if (get_broadcast_spec().m_type == op::BroadcastType::BIDIRECTIONAL)
    {
        auto arg_shape = inputs[0]->get_shape();
        Shape target_shape = op::util::BroadcastBase::get_target_shape(inputs[1]);
        PartialShape result_shape =
            get_result_shape_bidirectional(this, PartialShape{arg_shape}, target_shape);
        auto pair_broadcast_axes =
            get_broadcast_axes_bidirectional(arg_shape, result_shape.to_shape());
        return op::util::BroadcastBase::evaluate_broadcast(
            inputs[0], outputs[0], pair_broadcast_axes, result_shape.to_shape());
    }
    return op::util::BroadcastBase::evaluate(outputs, inputs);
}

void op::v3::Broadcast::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v3_Broadcast_validate_and_infer_types);
    if (m_mode.m_type == BroadcastType::NONE)
    {
        NODE_VALIDATION_CHECK(this,
                              get_input_size() == 3,
                              "axes_mapping input should be provided if explicit mode is used");
    }
    else
    {
        NODE_VALIDATION_CHECK(
            this,
            get_input_size() == 2,
            "axes_mapping input should not be provided for mode other than explicit");
    }

    util::BroadcastBase::validate_and_infer_types();

    auto result_shape = get_output_partial_shape(0);
    if (m_mode.m_type == BroadcastType::BIDIRECTIONAL)
    {
        if (get_input_partial_shape(0).rank().is_static() && get_input_partial_shape(1).is_static())
        {
            auto arg_shape = get_input_partial_shape(0);

            const auto shape_constant = get_constant_from_source(input_value(1));
            if (shape_constant)
            {
                auto target_shape = shape_constant->get_shape_val();
                result_shape = get_result_shape_bidirectional(this, arg_shape, target_shape);
            }
        }
    }
    set_input_is_relevant_to_shape(0); // arg - Result element type
    set_input_is_relevant_to_shape(1); // target_shape - Result shape
    if (get_input_size() == 3)
    {
        set_input_is_relevant_to_shape(2); // axes_mapping - Broadcast type
    }
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::v3::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_Broadcast_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2)
    {
        return make_shared<v3::Broadcast>(new_args.at(0), new_args.at(1), m_mode);
    }
    else if (new_args.size() == 3)
    {
        return make_shared<v3::Broadcast>(new_args.at(0), new_args.at(1), new_args.at(2), m_mode);
    }
    else
    {
        throw ngraph_error("Not supported number of Broadcast:v3 args");
    }
}

bool op::v3::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v3_Broadcast_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

bool op::v3::Broadcast::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v3_Broadcast_evaluate);
    return broadcast_evaluate(outputs, inputs);
}

bool op::v3::Broadcast::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v3_Broadcast_has_evaluate);
    return m_mode.m_type == BroadcastType::NONE || m_mode.m_type == BroadcastType::PDPD ||
           m_mode.m_type == BroadcastType::NUMPY || m_mode.m_type == BroadcastType::BIDIRECTIONAL;
}

namespace
{
    using namespace op;
    BroadcastModeSpec to_broadcast_mode(const AutoBroadcastSpec& bs)
    {
        BroadcastModeSpec broadcast_mode;
        broadcast_mode.m_axis = bs.m_axis;
        switch (bs.m_type)
        {
        case AutoBroadcastType::NONE: broadcast_mode.m_type = BroadcastType::NONE; break;
        case AutoBroadcastType::NUMPY: broadcast_mode.m_type = BroadcastType::NUMPY; break;
        case AutoBroadcastType::PDPD: broadcast_mode.m_type = BroadcastType::PDPD; break;
        }
        return broadcast_mode;
    }
} // namespace

constexpr NodeTypeInfo op::v1::Broadcast::type_info;

op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const Output<Node>& axes_mapping,
                             const AutoBroadcastSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, axes_mapping, to_broadcast_mode(broadcast_spec)}
    , m_broadcast_spec{broadcast_spec}
{
    constructor_validate_and_infer_types();
}

op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const AutoBroadcastSpec& broadcast_spec)
    : util::BroadcastBase{arg,
                          target_shape,
                          op::v0::Constant::create(element::u8, Shape{}, {0})->output(0),
                          to_broadcast_mode(broadcast_spec)}
    , m_broadcast_spec{broadcast_spec}
{
    constructor_validate_and_infer_types();
}

void op::v1::Broadcast::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v1_Broadcast_validate_and_infer_types);
    // m_type is deduced and not always explicitly stated, for cases where broadcast
    // has 2 inputs its always NUMPY mode
    if (m_broadcast_spec.m_type == AutoBroadcastType::NONE && get_input_size() < 3)
    {
        m_broadcast_spec.m_type = AutoBroadcastType::NUMPY;
    }

    // Mocking axes_mapping input for cases that don't require it
    if (m_broadcast_spec.m_type == AutoBroadcastType::NUMPY && get_input_size() < 3)
    {
        auto output = op::v0::Constant::create(element::u8, Shape{}, {0})->output(0);
        set_argument(2, output);
    }

    // update the base class' mode spec
    auto base_spec = to_broadcast_mode(m_broadcast_spec);
    if (util::BroadcastBase::m_mode.m_type != base_spec.m_type)
    {
        util::BroadcastBase::m_mode = base_spec;
    }

    util::BroadcastBase::validate_and_infer_types();
    set_input_is_relevant_to_shape(0); // arg - Result element type
    set_input_is_relevant_to_shape(1); // target_shape - Result shape
    set_input_is_relevant_to_shape(2); // axes_mapping - Broadcast type
}

shared_ptr<Node> op::v1::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v1_Broadcast_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Broadcast>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_broadcast_spec);
}

bool op::v1::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v1_Broadcast_visit_attributes);
    visitor.on_attribute("mode", m_broadcast_spec);
    return true;
}

bool op::v1::Broadcast::evaluate(const HostTensorVector& outputs,
                                 const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v1_Broadcast_evaluate);
    return op::util::BroadcastBase::evaluate(outputs, inputs);
}

bool op::v1::Broadcast::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v1_Broadcast_has_evaluate);
    return m_mode.m_type == BroadcastType::NONE || m_mode.m_type == BroadcastType::PDPD ||
           m_mode.m_type == BroadcastType::NUMPY;
}
