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

#include "broadcast_base.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/partial_shape.hpp"

#include "ngraph/runtime/reference/broadcast.hpp"

#include <numeric>

using namespace std;
using namespace ngraph;

op::util::BroadcastBase::BroadcastBase(const Output<Node>& arg,
                                       const Output<Node>& target_shape,
                                       const Output<Node>& axes_mapping,
                                       const BroadcastModeSpec& broadcast_mode)
    : Op({arg, target_shape, axes_mapping})
    , m_mode{broadcast_mode}
{
}

op::util::BroadcastBase::BroadcastBase(const Output<Node>& arg,
                                       const Output<Node>& target_shape,
                                       const BroadcastModeSpec& broadcast_mode)
    : Op({arg, target_shape})
    , m_mode{broadcast_mode}
{
}

PartialShape op::util::BroadcastBase::get_result_shape_pdpd(
    const PartialShape& arg0_shape,
    const Shape& target_shape,
    const op::BroadcastModeSpec& broadcast_spec) const
{
    if (arg0_shape.rank().is_dynamic())
    {
        return PartialShape::dynamic(target_shape.size());
    }
    const auto arg_rank_length = arg0_shape.rank().get_length();
    PartialShape result_shape = target_shape;
    auto start_axis = broadcast_spec.m_axis;

    NODE_VALIDATION_CHECK(this,
                          start_axis >= 0,
                          "Broadcast target_shape has smaller rank ",
                          target_shape.size(),
                          " than arg shape ",
                          arg_rank_length);
    for (auto i = start_axis; i < target_shape.size(); i++)
    {
        if (arg0_shape[i - start_axis].is_dynamic())
        {
            result_shape[i] = Dimension::dynamic();
            continue;
        }
        const size_t arg_dim = arg0_shape[i - start_axis].get_length();
        NODE_VALIDATION_CHECK(this,
                              arg_dim == 1 || target_shape[i] == 1 || arg_dim == target_shape[i],
                              "Broadcast incorrect target shape. Expecting either 1 or ",
                              arg_dim,
                              " . Got ",
                              target_shape[i]);
        result_shape[i] = std::max(arg_dim, target_shape[i]);
    }
    return result_shape;
}

void op::util::BroadcastBase::validate_target_shape_numpy(const PartialShape& arg_shape,
                                                          const Shape& target_shape) const
{
    if (arg_shape.rank().is_dynamic())
    {
        return;
    }
    const auto arg_rank_length = arg_shape.rank().get_length();
    auto start_axis = target_shape.size() - arg_rank_length;
    NODE_VALIDATION_CHECK(this,
                          start_axis >= 0,
                          "Broadcast target_shape has smaller rank ",
                          target_shape.size(),
                          " than arg shape ",
                          arg_rank_length);
    for (auto i = start_axis; i < target_shape.size(); i++)
    {
        if (arg_shape[i - start_axis].is_dynamic())
        {
            continue;
        }
        const size_t arg_dim = arg_shape[i - start_axis].get_length();
        NODE_VALIDATION_CHECK(this,
                              arg_dim == 1 || arg_dim == target_shape[i],
                              "Input shape dimension equal ",
                              arg_dim,
                              " cannot be broadcasted (numpy mode) to ",
                              target_shape[i],
                              ". Allowed input dimension value would be 1",
                              target_shape[i] != 1
                                  ? (std::string(" or ") + std::to_string(target_shape[i])).c_str()
                                  : "");
    }
}

void op::util::BroadcastBase::validate_target_shape_none(const Shape& arg_shape,
                                                         const AxisVector& axes_mapping_val,
                                                         const Shape& target_shape) const
{
    // axes_mapping needs to be in sorted order
    NODE_VALIDATION_CHECK(this,
                          std::is_sorted(axes_mapping_val.begin(), axes_mapping_val.end()),
                          "Broadcast doesn't permit transposes. axes_mapping ",
                          axes_mapping_val,
                          " not in sorted order");

    for (size_t i = 0; i < axes_mapping_val.size(); i++)
    {
        NODE_VALIDATION_CHECK(this,
                              axes_mapping_val[i] < target_shape.size(),
                              "Broadcast axes_mapping[",
                              i,
                              "]: ",
                              axes_mapping_val[i],
                              " exceeds target rank ",
                              target_shape.size());

        NODE_VALIDATION_CHECK(this,
                              target_shape[axes_mapping_val[i]] == arg_shape[i],
                              "Broadcast target[axes_mapping[",
                              i,
                              "]]",
                              " Expected ",
                              arg_shape[i],
                              ". Got ",
                              target_shape[axes_mapping_val[i]]);
    }
}

void op::util::BroadcastBase::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.is_integral_number(),
                          "Broadcast shape must be an integral number, but is: ",
                          shape_et);
    // shape node should produce a one dimensional shape.
    auto broadcast_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_rank.compatible(1),
                          "Broadcast shape rank must be 1, but has ",
                          broadcast_shape_rank);

    if (m_mode.m_type == BroadcastType::NONE)
    {
        // axes_mapping node should have integer data type. For now we only allow i64
        auto axes_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              axes_et.is_integral_number(),
                              "Broadcast axes must be integral numbers, but are: ",
                              axes_et);
        // axes_mapping node should produce a one dimensional shape.
        auto axes_shape_rank = get_input_partial_shape(2).rank();
        NODE_VALIDATION_CHECK(this,
                              axes_shape_rank.compatible(1),
                              "Broadcast axes rank must be 1, but has ",
                              axes_shape_rank);
    }

    PartialShape result_shape{PartialShape::dynamic()};
    const auto& input_shape = get_input_partial_shape(0);
    const auto input_rank = input_shape.rank();
    const auto& target_shape = input_value(1).get_partial_shape();
    const bool is_target_shape_known =
        target_shape.rank().is_static() && target_shape[0].is_static();

    if (m_mode.m_type == BroadcastType::BIDIRECTIONAL)
    {
        if (input_rank.is_static() && is_target_shape_known)
        {
            result_shape = PartialShape::dynamic(
                std::max(input_rank.get_length(), target_shape[0].get_length()));
        }
    }
    else
    {
        if (is_target_shape_known)
        {
            result_shape = PartialShape::dynamic(target_shape[0].get_length());
        }
    }

    const auto shape_constant = as_type_ptr<op::v0::Constant>(input_value(1).get_node_shared_ptr());

    if (auto concat = as_type_ptr<op::v0::Concat>(input_value(1).get_node_shared_ptr()))
    {
        auto concat_inputs = concat->inputs();

        if (concat->get_output_partial_shape(0).is_static() && concat->get_shape().size() == 1 &&
            concat_inputs.size() == shape_size(concat->get_shape()))
        {
            auto output_partial_shape = vector<Dimension>{};
            for (const auto& concat_input : concat_inputs)
            {
                auto source_node_ptr = concat_input.get_source_output().get_node_shared_ptr();
                if (auto source_const_ptr = as_type_ptr<op::v0::Constant>(source_node_ptr))
                {
                    output_partial_shape.push_back(source_const_ptr->get_axis_vector_val()[0]);
                }
                else
                {
                    output_partial_shape.push_back(Dimension::dynamic());
                }
            }
            result_shape = PartialShape(output_partial_shape);
        }
    }

    if (m_mode.m_type == BroadcastType::NONE)
    {
        if (shape_constant)
        {
            result_shape = shape_constant->get_shape_val();
        }
        // Validate axes_mapping
        if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static() &&
            get_input_partial_shape(2).is_static())
        {
            auto arg_shape = get_input_shape(0);
            auto axes_shape = get_input_shape(2);

            // Rank(arg_shape) == shape_size(axes_mapping)
            NODE_VALIDATION_CHECK(this,
                                  shape_size(axes_shape) == arg_shape.size(),
                                  "Broadcast axes_mapping shape ",
                                  axes_shape,
                                  " doesn't match rank of input tensor ",
                                  arg_shape.size());

            if (shape_constant && op::is_constant(input_value(2).get_node()))
            {
                auto target_shape = shape_constant->get_shape_val();
                auto axes_mapping_val =
                    as_type_ptr<op::v0::Constant>(input_value(2).get_node_shared_ptr())
                        ->get_axis_vector_val();
                validate_target_shape_none(arg_shape, axes_mapping_val, target_shape);
            }
        }
    }
    else if (m_mode.m_type == BroadcastType::NUMPY)
    {
        if (shape_constant)
        {
            const auto target_shape = shape_constant->get_shape_val();
            result_shape = target_shape;
            validate_target_shape_numpy(input_shape, target_shape);
        }
    }
    else if (m_mode.m_type == BroadcastType::PDPD)
    {
        if (shape_constant)
        {
            const auto target_shape = shape_constant->get_shape_val();
            result_shape = get_result_shape_pdpd(input_shape, target_shape, m_mode);
        }
    }
    set_output_type(0, get_input_element_type(0), result_shape);
}

std::pair<bool, AxisSet> op::util::BroadcastBase::get_broadcast_axes_numpy_pdpd(
    const Shape& arg_shape, const Shape& result_shape, const op::BroadcastModeSpec& broadcast_spec)
{
    AxisSet broadcast_axes;
    bool axes_known = false;
    auto start_axis = (broadcast_spec.m_type == op::BroadcastType::PDPD)
                          ? broadcast_spec.m_axis
                          : result_shape.size() - arg_shape.size();
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

std::pair<bool, AxisSet>
    op::util::BroadcastBase::get_broadcast_axes_none(const AxisVector axes_mapping_val,
                                                     const size_t target_shape_size)
{
    AxisSet broadcast_axes;
    bool axes_known = false;

    std::vector<size_t> axes(target_shape_size);
    std::iota(axes.begin(), axes.end(), 0);
    for (auto i = axes_mapping_val.rbegin(); i != axes_mapping_val.rend(); ++i)
    {
        axes.erase(axes.begin() + *i);
    }
    broadcast_axes.insert(axes.begin(), axes.end());

    axes_known = true;
    return std::make_pair(axes_known, broadcast_axes);
}

std::pair<bool, AxisSet> op::util::BroadcastBase::get_broadcast_axes() const
{
    AxisSet broadcast_axes;
    bool axes_known = false;

    if (m_mode.m_type == BroadcastType::NONE)
    {
        const auto axes_mapping_constant =
            as_type_ptr<op::v0::Constant>(input_value(2).get_node_shared_ptr());
        if (get_input_partial_shape(1).is_static() && axes_mapping_constant)
        {
            auto axes_mapping_val = axes_mapping_constant->get_axis_vector_val();
            auto target_shape = get_input_shape(1);
            NGRAPH_CHECK(target_shape.size() == 1);
            return get_broadcast_axes_none(axes_mapping_val, target_shape[0]);
        }
    }
    else if (m_mode.m_type == BroadcastType::NUMPY || m_mode.m_type == BroadcastType::PDPD)
    {
        if (get_input_partial_shape(0).is_static() && get_output_partial_shape(0).is_static())
        {
            auto arg_shape = get_input_shape(0);
            auto result_shape = get_output_shape(0);
            return get_broadcast_axes_numpy_pdpd(arg_shape, result_shape, m_mode);
        }
    }
    else
    {
        throw ngraph_error("Unknown autobroadcast type");
    }

    return std::make_pair(axes_known, broadcast_axes);
}

template <element::Type_t ET>
bool op::util::BroadcastBase::evaluate(const HostTensorPtr& arg0,
                                       const HostTensorPtr& out,
                                       const AxisSet& broadcast_axes) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::util::BroadcastBase::evaluate<ET>");
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::broadcast<T>((arg0->get_data_ptr<ET>()),
                                     (out->get_data_ptr<ET>()),
                                     arg0->get_shape(),
                                     out->get_shape(),
                                     broadcast_axes);
    return true;
}

namespace
{
    template <element::Type_t ET>
    void get_axis_vector_from_hosttensor(const HostTensorPtr& arg, AxisVector& axes_vector)
    {
        using T = typename element_type_traits<ET>::value_type;
        auto rank = arg->get_shape().at(0);
        std::vector<T> axes_vec(rank);
        arg->read(axes_vec.data(), rank * sizeof(T));
        axes_vector = AxisVector(axes_vec.begin(), axes_vec.end());
    }

#define GET_AXIS_VECTOR(a)                                                                         \
    case element::Type_t::a: get_axis_vector_from_hosttensor<element::Type_t::a>

    void get_axis_vector_from_ht(const HostTensorPtr& arg,
                                 AxisVector& axis_vector,
                                 const Shape& arg_shape)
    {
        switch (arg->get_element_type())
        {
            GET_AXIS_VECTOR(i8)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(i16)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(i32)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(i64)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(u8)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(u16)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(u32)(arg, axis_vector);
            break;
            GET_AXIS_VECTOR(u64)(arg, axis_vector);
            break;
        default:
            // other types are not supported and would have thrown in ctor
            ngraph_error("get_axis_vector_from_ht: type is not integral\n");
            break;
        }
        // Rank(arg_shape) == shape_size(axes_mapping)
        NGRAPH_CHECK(axis_vector.size() == arg_shape.size(),
                     "Broadcast axes_mapping shape ",
                     axis_vector.size(),
                     " doesn't match rank of input tensor ",
                     arg_shape.size());
    }

    template <element::Type_t ET>
    void get_shape_from_hosttensor(const HostTensorPtr& input1, Shape& target_shape)
    {
        using T = typename element_type_traits<ET>::value_type;
        auto rank = input1->get_shape().at(0);
        std::vector<T> target_shape_vec(rank);
        input1->read(target_shape_vec.data(), rank * sizeof(T));
        target_shape = Shape(target_shape_vec.begin(), target_shape_vec.end());
    }

#define CASE_GET_SHAPE(a)                                                                          \
    case element::Type_t::a: get_shape_from_hosttensor<element::Type_t::a>

    Shape get_target_shape_from_ht(const HostTensorPtr& input1)
    {
        Shape target_shape;
        switch (input1->get_element_type())
        {
            CASE_GET_SHAPE(i8)(input1, target_shape);
            break;
            CASE_GET_SHAPE(i16)(input1, target_shape);
            break;
            CASE_GET_SHAPE(i32)(input1, target_shape);
            break;
            CASE_GET_SHAPE(i64)(input1, target_shape);
            break;
            CASE_GET_SHAPE(u8)(input1, target_shape);
            break;
            CASE_GET_SHAPE(u16)(input1, target_shape);
            break;
            CASE_GET_SHAPE(u32)(input1, target_shape);
            break;
            CASE_GET_SHAPE(u64)(input1, target_shape);
            break;
        default:
            // other types are not supported and would have thrown in ctor
            ngraph_error("get_target_shape_from_ht: type is not integral\n");
            break;
        }
        return target_shape;
    }
}

bool op::util::BroadcastBase::evaluate_broadcast(const HostTensorPtr& arg0,
                                                 const HostTensorPtr& out,
                                                 const std::pair<bool, AxisSet> pair_broadcast_axes,
                                                 const Shape output_shape) const
{
    if (!pair_broadcast_axes.first)
    {
        // broadcast_axes not known deterministically
        return false;
    }
    bool rc = true;
    Shape in_shape = arg0->get_shape();
    out->set_shape(output_shape);
    out->set_element_type(arg0->get_element_type());
    switch (arg0->get_element_type())
    {
        TYPE_CASE(boolean)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(i8)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(i16)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(i32)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(i64)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(u8)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(u16)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(u32)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(u64)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(f16)(arg0, out, pair_broadcast_axes.second);
        break;
        TYPE_CASE(f32)(arg0, out, pair_broadcast_axes.second);
        break;
    default: rc = false; break;
    }
    return rc;
}

Shape op::util::BroadcastBase::get_target_shape(const HostTensorPtr& input1) const
{
    Shape target_shape;
    const auto shape_constant = as_type_ptr<op::v0::Constant>(input_value(1).get_node_shared_ptr());
    if (shape_constant)
    {
        target_shape = shape_constant->get_shape_val();
    }
    else
    {
        target_shape = get_target_shape_from_ht(input1);
    }
    return target_shape;
}

bool op::util::BroadcastBase::evaluate(const HostTensorVector& outputs,
                                       const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::util::BroadcastBase::evaluate");

    Shape target_shape = get_target_shape(inputs[1]);

    PartialShape result_shape;
    std::pair<bool, AxisSet> pair_broadcast_axes;
    auto arg_shape = inputs[0]->get_shape();

    if (m_mode.m_type == BroadcastType::NONE)
    {
        AxisVector axes_mapping_val;
        const auto axes_mapping_constant =
            as_type_ptr<op::v0::Constant>(input_value(2).get_node_shared_ptr());
        if (axes_mapping_constant)
        {
            axes_mapping_val = axes_mapping_constant->get_axis_vector_val();
        }
        else
        {
            // read from HT and save as AxisVector
            get_axis_vector_from_ht(inputs[2], axes_mapping_val, arg_shape);
        }
        pair_broadcast_axes = get_broadcast_axes_none(axes_mapping_val, target_shape.size());
        validate_target_shape_none(inputs[0]->get_shape(), axes_mapping_val, target_shape);
        result_shape = target_shape;
    }
    else if (m_mode.m_type == BroadcastType::PDPD)
    {
        result_shape = get_result_shape_pdpd(arg_shape, target_shape, m_mode);
        pair_broadcast_axes =
            get_broadcast_axes_numpy_pdpd(arg_shape, result_shape.to_shape(), m_mode);
    }
    else if (m_mode.m_type == BroadcastType::NUMPY)
    {
        result_shape = target_shape;
        validate_target_shape_numpy(arg_shape, target_shape);
        pair_broadcast_axes =
            get_broadcast_axes_numpy_pdpd(arg_shape, result_shape.to_shape(), m_mode);
    }
    else
    {
        ngraph_error("Unsupported BroadcastType ");
    }

    return evaluate_broadcast(inputs[0], outputs[0], pair_broadcast_axes, result_shape.to_shape());
}
