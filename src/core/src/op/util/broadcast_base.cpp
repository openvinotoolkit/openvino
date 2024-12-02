// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/broadcast_base.hpp"

#include <numeric>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/broadcast.hpp"

ov::op::util::BroadcastBase::BroadcastBase(const Output<Node>& arg,
                                           const Output<Node>& target_shape,
                                           const Output<Node>& axes_mapping,
                                           const BroadcastModeSpec& broadcast_mode)
    : Op({arg, target_shape, axes_mapping}),
      m_mode{broadcast_mode} {
    ov::mark_as_precision_sensitive(input(1));
}

ov::op::util::BroadcastBase::BroadcastBase(const Output<Node>& arg,
                                           const Output<Node>& target_shape,
                                           const BroadcastModeSpec& broadcast_mode)
    : Op({arg, target_shape}),
      m_mode{broadcast_mode} {
    ov::mark_as_precision_sensitive(input(1));
}

ov::PartialShape ov::op::util::BroadcastBase::get_result_shape_pdpd(const PartialShape& arg0_shape,
                                                                    const PartialShape& target_pshape,
                                                                    const op::BroadcastModeSpec& broadcast_spec) const {
    if (target_pshape.is_dynamic())
        return PartialShape::dynamic(target_pshape.rank());
    Shape target_shape = target_pshape.to_shape();
    if (arg0_shape.rank().is_dynamic()) {
        return PartialShape::dynamic(target_shape.size());
    }
    const auto arg_rank_length = arg0_shape.rank().get_length();
    PartialShape result_shape = target_shape;
    auto start_axis = ((broadcast_spec.m_type == op::BroadcastType::PDPD) && (broadcast_spec.m_axis == -1))
                          ? static_cast<int64_t>(target_pshape.size()) - static_cast<int64_t>(arg0_shape.size())
                          : broadcast_spec.m_axis;

    NODE_VALIDATION_CHECK(this,
                          start_axis >= 0,
                          "Broadcast target_shape has smaller rank ",
                          target_shape.size(),
                          " than arg shape ",
                          arg_rank_length);
    for (size_t i = start_axis; i < target_shape.size(); i++) {
        if (arg0_shape[i - start_axis].is_dynamic()) {
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

void ov::op::util::BroadcastBase::validate_target_shape_numpy(const PartialShape& arg_shape,
                                                              const PartialShape& target_shape) const {
    if (arg_shape.rank().is_dynamic() || target_shape.rank().is_dynamic()) {
        return;
    }
    const auto arg_rank_length = arg_shape.rank().get_length();
    const auto target_rank_length = target_shape.rank().get_length();
    const int64_t start_axis = target_rank_length - arg_rank_length;
    NODE_VALIDATION_CHECK(this,
                          start_axis >= 0,
                          "Broadcast target_shape has smaller rank ",
                          target_rank_length,
                          " than arg shape ",
                          arg_rank_length);
    for (auto i = start_axis; i < target_rank_length; i++) {
        std::stringstream ss;
        ss << " or " << target_shape[i];
        NODE_VALIDATION_CHECK(this,
                              arg_shape[i - start_axis].is_dynamic() || target_shape[i].is_dynamic() ||
                                  arg_shape[i - start_axis] == 1 || arg_shape[i - start_axis] == target_shape[i],
                              "Input shape dimension equal ",
                              arg_shape[i - start_axis],
                              " cannot be broadcasted (numpy mode) to ",
                              target_shape[i],
                              ". Allowed input dimension value would be 1",
                              target_shape[i] != 1 ? ss.str() : "");
    }
}

void ov::op::util::BroadcastBase::validate_target_shape_none(const PartialShape& arg_shape,
                                                             const AxisVector& axes_mapping_val,
                                                             const PartialShape& target_shape) const {
    if (arg_shape.rank().is_dynamic() || target_shape.rank().is_dynamic()) {
        return;
    }
    const auto target_rank_length = target_shape.rank().get_length();
    // axes_mapping needs to be in sorted order
    NODE_VALIDATION_CHECK(this,
                          std::is_sorted(axes_mapping_val.begin(), axes_mapping_val.end()),
                          "Broadcast doesn't permit transposes. axes_mapping ",
                          axes_mapping_val,
                          " not in sorted order");

    if (arg_shape.rank().get_length() == 0 && axes_mapping_val.size() > 0) {
        NODE_VALIDATION_CHECK(this,
                              target_shape[axes_mapping_val[0]].compatible(1),
                              "Broadcast target[axes_mapping[0]]. Expected 1. Got ",
                              target_shape[axes_mapping_val[0]]);
    }

    for (size_t i = 0; i < axes_mapping_val.size(); i++) {
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(axes_mapping_val[i]) < target_rank_length,
                              "Broadcast axes_mapping[",
                              i,
                              "]: ",
                              axes_mapping_val[i],
                              " exceeds target rank ",
                              target_rank_length);

        if (arg_shape.rank().get_length() > 0) {
            NODE_VALIDATION_CHECK(
                this,
                target_shape[axes_mapping_val[i]].compatible(arg_shape[i]) || arg_shape[i].compatible(1),
                "Broadcast target[axes_mapping[",
                i,
                "]]",
                " Expected ",
                arg_shape[i],
                ". Got ",
                target_shape[axes_mapping_val[i]]);
        }
    }
}

void ov::op::util::BroadcastBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_BroadcastBase_validate_and_infer_types);
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

    if (m_mode.m_type == BroadcastType::NONE) {
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
    const bool is_target_shape_known = target_shape.rank().is_static() && target_shape[0].is_static();

    if (m_mode.m_type == BroadcastType::BIDIRECTIONAL) {
        if (input_rank.is_static() && is_target_shape_known) {
            result_shape = PartialShape::dynamic(std::max(input_rank.get_length(), target_shape[0].get_length()));
        }
    } else {
        if (is_target_shape_known) {
            result_shape = PartialShape::dynamic(target_shape[0].get_length());
        }
    }

    PartialShape output_shape;
    bool output_shape_defined = ov::util::evaluate_as_partial_shape(get_input_source_output(1), output_shape);

    if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(input_value(1).get_node_shared_ptr())) {
        auto concat_inputs = concat->inputs();

        if (!output_shape_defined && concat->get_output_partial_shape(0).is_static() &&
            concat->get_shape().size() == 1 && concat_inputs.size() == shape_size(concat->get_shape())) {
            output_shape.resize(0);
            for (const auto& concat_input : concat_inputs) {
                auto source_node_ptr = concat_input.get_source_output().get_node_shared_ptr();
                if (auto source_const_ptr = ov::as_type_ptr<ov::op::v0::Constant>(source_node_ptr)) {
                    output_shape.emplace_back(source_const_ptr->get_axis_vector_val()[0]);
                } else {
                    output_shape.push_back(Dimension::dynamic());
                }
            }
            output_shape_defined = true;
        }
    }

    if (m_mode.m_type == BroadcastType::NONE) {
        if (output_shape_defined) {
            result_shape = output_shape;
        }
        // Validate axes_mapping
        if (get_input_partial_shape(0).is_static() && get_input_partial_shape(1).is_static() &&
            get_input_partial_shape(2).is_static()) {
            const auto& arg_shape = get_input_shape(0);
            const auto& axes_shape = get_input_shape(2);
            auto input_rank = (arg_shape.size() == 0 && shape_size(axes_shape) > 0) ? 1 : arg_shape.size();

            // Rank(arg_shape) == shape_size(axes_mapping)
            NODE_VALIDATION_CHECK(this,
                                  shape_size(axes_shape) == input_rank,
                                  "Broadcast axes_mapping shape ",
                                  axes_shape,
                                  " doesn't match rank of input tensor ",
                                  input_rank);

            if (output_shape_defined && has_and_set_equal_bounds(input_value(2))) {
                auto axes_mapping_val = ov::util::get_constant_from_source(input_value(2))->get_axis_vector_val();
                validate_target_shape_none(arg_shape, axes_mapping_val, output_shape);
            }
        }
    } else if (m_mode.m_type == BroadcastType::NUMPY) {
        if (output_shape_defined) {
            result_shape = output_shape;
            validate_target_shape_numpy(input_shape, output_shape);
        }
    } else if (m_mode.m_type == BroadcastType::PDPD) {
        if (output_shape_defined) {
            result_shape = get_result_shape_pdpd(input_shape, output_shape, m_mode);
        }
    }
    set_output_type(0, get_input_element_type(0), result_shape);
}

std::pair<bool, ov::AxisSet> ov::op::util::BroadcastBase::get_broadcast_axes_numpy_pdpd(
    const Shape& arg_shape,
    const Shape& result_shape,
    const op::BroadcastModeSpec& broadcast_spec) {
    AxisSet broadcast_axes;
    bool axes_known = false;
    int64_t start_axis = ((broadcast_spec.m_type == op::BroadcastType::PDPD) && (broadcast_spec.m_axis != -1))
                             ? broadcast_spec.m_axis
                             : static_cast<int64_t>(result_shape.size()) - static_cast<int64_t>(arg_shape.size());
    OPENVINO_ASSERT(start_axis >= 0);
    for (size_t i = 0; i < result_shape.size(); i++) {
        if (i < static_cast<size_t>(start_axis) || result_shape[i] != arg_shape[i - start_axis]) {
            broadcast_axes.insert(i);
        }
    }
    axes_known = true;
    return std::make_pair(axes_known, broadcast_axes);
}

std::pair<bool, ov::AxisSet> ov::op::util::BroadcastBase::get_broadcast_axes_none(const AxisVector& axes_mapping_val,
                                                                                  const size_t target_shape_size) {
    AxisSet broadcast_axes;
    bool axes_known = false;

    std::vector<size_t> axes(target_shape_size);
    std::iota(axes.begin(), axes.end(), 0);
    for (auto i = axes_mapping_val.rbegin(); i != axes_mapping_val.rend(); ++i) {
        axes.erase(axes.begin() + *i);
    }
    broadcast_axes.insert(axes.begin(), axes.end());

    axes_known = true;
    return std::make_pair(axes_known, broadcast_axes);
}

std::pair<bool, ov::AxisSet> ov::op::util::BroadcastBase::get_broadcast_axes() const {
    AxisSet broadcast_axes;
    bool axes_known = false;

    if (m_mode.m_type == BroadcastType::NONE) {
        const auto axes_mapping_constant = ov::util::get_constant_from_source(input_value(2));
        if (get_input_partial_shape(1).is_static() && axes_mapping_constant) {
            auto axes_mapping_val = axes_mapping_constant->get_axis_vector_val();
            auto target_shape = get_input_shape(1);
            OPENVINO_ASSERT(target_shape.size() == 1);
            return get_broadcast_axes_none(axes_mapping_val, target_shape[0]);
        }
    } else if (m_mode.m_type == BroadcastType::NUMPY || m_mode.m_type == BroadcastType::PDPD) {
        if (get_input_partial_shape(0).is_static() && get_output_partial_shape(0).is_static()) {
            const auto& arg_shape = get_input_shape(0);
            const auto& result_shape = get_output_shape(0);
            return get_broadcast_axes_numpy_pdpd(arg_shape, result_shape, m_mode);
        }
    } else {
        OPENVINO_THROW("Unknown autobroadcast type");
    }

    return std::make_pair(axes_known, broadcast_axes);
}

bool ov::op::util::BroadcastBase::evaluate_broadcast(const ov::Tensor& arg0,
                                                     ov::Tensor& out,
                                                     const AxisSet& broadcast_axes) const {
    OV_OP_SCOPE(util_BroadcastBase_evaluate_axes);
    auto arg0_shape = arg0.get_shape();
    if (arg0_shape.size() == 0) {
        arg0_shape = Shape{1};
    }
    ov::reference::broadcast(static_cast<const char*>(arg0.data()),
                             static_cast<char*>(out.data()),
                             arg0_shape,
                             out.get_shape(),
                             broadcast_axes,
                             arg0.get_element_type().size());
    return true;
}

namespace {
template <ov::element::Type_t ET>
void get_axis_vector_from_hosttensor(const ov::Tensor& arg, ov::AxisVector& axes_vector) {
    using T = typename ov::element_type_traits<ET>::value_type;
    auto rank = arg.get_shape().at(0);
    std::vector<T> axes_vec(rank);
    std::memcpy(axes_vec.data(), arg.data(), rank * sizeof(T));
    axes_vector = ov::AxisVector(axes_vec.begin(), axes_vec.end());
}

#define GET_AXIS_VECTOR(a)       \
    case ov::element::Type_t::a: \
        get_axis_vector_from_hosttensor<ov::element::Type_t::a>

void get_axis_vector_from_ht(const ov::Tensor& arg, ov::AxisVector& axis_vector, const ov::Shape& arg_shape) {
    switch (arg.get_element_type()) {
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
        OPENVINO_THROW("get_axis_vector_from_ht: type is not integral");
    }
    // Rank(arg_shape) == shape_size(axes_mapping)
    OPENVINO_ASSERT(axis_vector.size() == arg_shape.size(),
                    "Broadcast axes_mapping shape ",
                    axis_vector.size(),
                    " doesn't match rank of input tensor ",
                    arg_shape.size());
}

template <ov::element::Type_t ET>
void get_shape_from_hosttensor(const ov::Tensor& input1, ov::Shape& target_shape) {
    using T = typename ov::element_type_traits<ET>::value_type;
    auto rank = input1.get_shape().at(0);
    std::vector<T> target_shape_vec(rank);
    std::memcpy(target_shape_vec.data(), input1.data(), rank * sizeof(T));
    target_shape = ov::Shape(target_shape_vec.begin(), target_shape_vec.end());
}

#define CASE_GET_SHAPE(a)        \
    case ov::element::Type_t::a: \
        get_shape_from_hosttensor<ov::element::Type_t::a>

ov::Shape get_target_shape_from_ht(const ov::Tensor& input1) {
    ov::Shape target_shape;
    switch (input1.get_element_type()) {
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
        OPENVINO_THROW("get_target_shape_from_ht: type is not integral");
    }
    return target_shape;
}
}  // namespace

bool ov::op::util::BroadcastBase::evaluate_broadcast(const ov::Tensor& arg0,
                                                     ov::Tensor& out,
                                                     const std::pair<bool, AxisSet>& pair_broadcast_axes,
                                                     const Shape& output_shape) const {
    if (!pair_broadcast_axes.first) {
        // broadcast_axes not known deterministically
        return false;
    }
    Shape in_shape = arg0.get_shape();
    out.set_shape(output_shape);

    return evaluate_broadcast(arg0, out, pair_broadcast_axes.second);
}

ov::Shape ov::op::util::BroadcastBase::get_target_shape(const ov::Tensor& input1) const {
    return get_target_shape_from_ht(input1);
}

bool ov::op::util::BroadcastBase::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(util_BroadcastBase_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    OPENVINO_ASSERT(outputs.size(), 1);
    Shape target_shape = get_target_shape(inputs[1]);

    PartialShape result_shape;
    std::pair<bool, AxisSet> pair_broadcast_axes;
    const auto& arg_shape = inputs[0].get_shape();

    if (m_mode.m_type == BroadcastType::NONE) {
        AxisVector axes_mapping_val;
        // read from HT and save as AxisVector
        get_axis_vector_from_ht(inputs[2], axes_mapping_val, arg_shape);

        pair_broadcast_axes = get_broadcast_axes_none(axes_mapping_val, target_shape.size());
        validate_target_shape_none(inputs[0].get_shape(), axes_mapping_val, target_shape);
        result_shape = target_shape;
    } else if (m_mode.m_type == BroadcastType::PDPD) {
        result_shape = get_result_shape_pdpd(arg_shape, target_shape, m_mode);
        pair_broadcast_axes = get_broadcast_axes_numpy_pdpd(arg_shape, result_shape.to_shape(), m_mode);
    } else if (m_mode.m_type == BroadcastType::NUMPY) {
        result_shape = target_shape;
        validate_target_shape_numpy(arg_shape, target_shape);
        pair_broadcast_axes = get_broadcast_axes_numpy_pdpd(arg_shape, result_shape.to_shape(), m_mode);
    } else {
        OPENVINO_THROW("Unsupported BroadcastType ");
    }

    return evaluate_broadcast(inputs[0], outputs[0], pair_broadcast_axes, result_shape.to_shape());
}

bool ov::op::util::BroadcastBase::evaluate_lower(ov::TensorVector& output_values) const {
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        (get_input_size() > 2 && !input_value(2).get_tensor().has_and_set_bound()))
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool ov::op::util::BroadcastBase::evaluate_upper(ov::TensorVector& output_values) const {
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        (get_input_size() > 2 && !input_value(2).get_tensor().has_and_set_bound()))
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool ov::op::util::BroadcastBase::evaluate_symbol(ov::TensorSymbolVector& output_symbols) const {
    if (!input_value(1).get_tensor().has_and_set_bound() ||
        (get_input_size() > 2 && !input_value(2).get_tensor().has_and_set_bound()))
        return false;
    return default_symbol_evaluator(this, {0}, output_symbols);
}
