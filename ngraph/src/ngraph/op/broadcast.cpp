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

#include "ngraph/op/broadcast.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/partial_shape.hpp"

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
}

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
    PartialShape get_result_shape_bidirectional(Node* this_ptr,
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
        for (auto i = 0; i < target_shape.size(); ++i)
        {
            if (arg_shape_vec[i].is_dynamic())
            {
                result_shape[i] = Dimension::dynamic();
                continue;
            }
            const size_t arg_shape_dim = arg_shape_vec[i].get_length();
            NODE_VALIDATION_CHECK(this_ptr,
                                  arg_shape_dim == 1 || target_shape[i] == 1 ||
                                      arg_shape[i] == target_shape[i],
                                  "Broadcast incorrect target shape. Expecting either 1 or ",
                                  arg_shape_dim,
                                  ". Got ",
                                  target_shape[i]);

            result_shape[i] = std::max(arg_shape_dim, target_shape[i]);
        }
        return result_shape;
    }
}

void op::v3::Broadcast::validate_and_infer_types()
{
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

            const auto shape_constant =
                as_type_ptr<op::v0::Constant>(input_value(1).get_node_shared_ptr());
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
    visitor.on_attribute("broadcast_spec", m_mode);
    return true;
}

bool op::v3::Broadcast::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
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
}

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
    util::BroadcastBase::validate_and_infer_types();

    set_input_is_relevant_to_shape(0); // arg - Result element type
    set_input_is_relevant_to_shape(1); // target_shape - Result shape
    set_input_is_relevant_to_shape(2); // axes_mapping - Broadcast type
}

shared_ptr<Node> op::v1::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Broadcast>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_broadcast_spec);
}

bool op::v1::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("broadcast_spec", m_broadcast_spec);
    return true;
}

bool op::v1::Broadcast::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return op::util::BroadcastBase::evaluate(outputs, inputs);
}

constexpr NodeTypeInfo op::v0::Broadcast::type_info;

op::v0::Broadcast::Broadcast(const OutputVector& args,
                             const Shape& shape,
                             const AxisSet& broadcast_axes)
    : Op(args)
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
    constructor_validate_and_infer_types();
}

op::v0::Broadcast::Broadcast(const Output<Node>& arg,
                             const Shape& shape,
                             const AxisSet& broadcast_axes)
    : Broadcast(OutputVector{arg}, shape, broadcast_axes)
{
}

bool op::v0::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("shape", m_shape);
    visitor.on_attribute("broadcast_axes", m_broadcast_axes);
    return true;
}

void op::v0::Broadcast::validate_and_infer_types()
{
    infer_shape();

    for (auto axis : m_broadcast_axes)
    {
        NODE_VALIDATION_CHECK(this,
                              axis < m_shape.size(),
                              "Broadcast axis index (",
                              axis,
                              ") exceeds specified output shape rank ",
                              "(broadcast axes: ",
                              m_broadcast_axes,
                              ", output shape: ",
                              m_shape,
                              ").");
    }

    Shape required_input_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        required_input_shape.erase(required_input_shape.begin() + *i);
    }

    // TODO(amprocte): We can probably have a more helpful error message here.
    // There are two things that can go wrong, which are being picked up in
    // one fell swoop by this check: either the number of broadcast axes is not
    // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(0).compatible(required_input_shape),
        "Broadcast argument shape, specified output shape, and axes are incompatible ",
        "(argument shape: ",
        get_input_partial_shape(0),
        ", output shape: ",
        m_shape,
        ", broadcast axes: ",
        m_broadcast_axes,
        ").");

    set_output_type(0, get_input_element_type(0), m_shape);
}

shared_ptr<Node> op::v0::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
}

void op::v0::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, m_broadcast_axes));
}

namespace
{
#define TYPE_CASE_v0(a)                                                                            \
    case element::Type_t::a: rc = evaluate_v0<element::Type_t::a>

    template <element::Type_t ET>
    inline bool evaluate_v0(const HostTensorPtr& arg0,
                            const HostTensorPtr& out,
                            const AxisSet& broadcast_axes)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::broadcast<T>((arg0->get_data_ptr<ET>()),
                                         (out->get_data_ptr<ET>()),
                                         arg0->get_shape(),
                                         out->get_shape(),
                                         broadcast_axes);
        return true;
    }

    bool evaluate_broadcast_v0(const HostTensorPtr& arg0,
                               const HostTensorPtr& out,
                               const AxisSet broadcast_axes,
                               const Shape output_shape)
    {
        bool rc = true;
        Shape in_shape = arg0->get_shape();
        out->set_shape(output_shape);
        out->set_element_type(arg0->get_element_type());
        switch (arg0->get_element_type())
        {
            TYPE_CASE_v0(boolean)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(i8)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(i16)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(i32)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(i64)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(u8)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(u16)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(u32)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(u64)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(bf16)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(f16)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(f32)(arg0, out, broadcast_axes);
            break;
            TYPE_CASE_v0(f64)(arg0, out, broadcast_axes);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::v0::Broadcast::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    return evaluate_broadcast_v0(inputs[0], outputs[0], get_broadcast_axes(), get_output_shape(0));
}

constexpr NodeTypeInfo op::v0::BroadcastLike::type_info;

op::v0::BroadcastLike::BroadcastLike(const Output<Node>& arg,
                                     const Output<Node>& like_arg,
                                     const AxisSet& initial_broadcast_axes)
    : op::v0::Broadcast({arg, like_arg}, {}, {})
    , m_initial_broadcast_axes(initial_broadcast_axes)
{
    constructor_validate_and_infer_types();
}

bool op::v0::BroadcastLike::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("shape", m_shape);
    visitor.on_attribute("broadcast_axes", m_broadcast_axes);
    visitor.on_attribute("initial_broadcast_axes", m_initial_broadcast_axes);
    return true;
}

shared_ptr<Node> op::v0::BroadcastLike::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<v0::BroadcastLike>(new_args.at(0), new_args.at(1), m_initial_broadcast_axes);
}

void op::v0::BroadcastLike::infer_shape()
{
    const Shape& in_shape = get_input_shape(0);
    m_shape = get_input_shape(1);
    m_broadcast_axes = m_initial_broadcast_axes;
    if (m_broadcast_axes.size() == 0)
    {
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            if (i < in_shape.size())
            {
                if (in_shape.at(i) == 1 && m_shape.at(i) > 1)
                {
                    m_broadcast_axes.insert(i);
                }
            }
            else
            {
                m_broadcast_axes.insert(i);
            }
        }
    }
}
