// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/broadcast.hpp"

#include <broadcast_shape_inference.hpp>
#include <numeric>

#include "itt.hpp"
#include "openvino/reference/broadcast.hpp"

ov::op::v3::Broadcast::Broadcast(const Output<Node>& arg,
                                 const Output<Node>& target_shape,
                                 const Output<Node>& axes_mapping,
                                 const BroadcastModeSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, axes_mapping, broadcast_spec} {
    constructor_validate_and_infer_types();
}

ov::op::v3::Broadcast::Broadcast(const Output<Node>& arg,
                                 const Output<Node>& target_shape,
                                 const BroadcastModeSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, broadcast_spec} {
    constructor_validate_and_infer_types();
}

namespace {
std::pair<bool, ov::AxisSet> get_broadcast_axes_bidirectional(const ov::Shape& arg_shape,
                                                              const ov::Shape& result_shape) {
    ov::AxisSet broadcast_axes;
    bool axes_known = false;
    const auto start_axis = static_cast<int64_t>(result_shape.size()) - static_cast<int64_t>(arg_shape.size());
    OPENVINO_ASSERT(start_axis >= 0);
    for (size_t i = 0; i < result_shape.size(); i++) {
        if (i < static_cast<size_t>(start_axis) || result_shape[i] != arg_shape[i - start_axis]) {
            broadcast_axes.insert(i);
        }
    }
    axes_known = true;
    return std::make_pair(axes_known, broadcast_axes);
}
}  // namespace

std::pair<bool, ov::AxisSet> ov::op::v3::Broadcast::get_broadcast_axes() const {
    if (m_mode.m_type == BroadcastType::BIDIRECTIONAL) {
        AxisSet broadcast_axes;
        bool axes_known = false;

        if (get_input_partial_shape(0).is_static() && get_output_partial_shape(0).is_static()) {
            const auto& arg_shape = get_input_shape(0);
            const auto& result_shape = get_output_shape(0);
            return get_broadcast_axes_bidirectional(arg_shape, result_shape);
        }
        return std::make_pair(axes_known, broadcast_axes);
    }

    return util::BroadcastBase::get_broadcast_axes();
}

namespace {
ov::PartialShape get_result_shape_bidirectional(const ov::Node* this_ptr,
                                                ov::PartialShape arg_shape,
                                                ov::PartialShape target_shape) {
    if (arg_shape.rank().is_dynamic() || target_shape.rank().is_dynamic()) {
        return ov::PartialShape::dynamic();
    }
    ov::PartialShape result_shape;
    // Add left padding to shorter target or argument shape
    const auto target_padded_rank = std::max(arg_shape.size(), target_shape.size());
    while (arg_shape.size() < target_padded_rank) {
        arg_shape.insert(arg_shape.begin(), 1);
    }
    while (target_shape.size() < target_padded_rank) {
        target_shape.insert(target_shape.begin(), 1);
    }

    result_shape = target_shape;
    for (size_t i = 0; i < target_shape.size(); ++i) {
        const auto& arg_dim = arg_shape[i];
        const auto& target_dim = target_shape[i];

        if (arg_dim.is_dynamic() || target_dim.is_dynamic()) {
            if (target_dim == 1 || (arg_dim.is_static() && arg_dim != 1)) {
                result_shape[i] = arg_dim;
            } else if (arg_dim == 1 || (target_dim.is_static() && target_dim != 1)) {
                result_shape[i] = target_dim;
            } else {
                result_shape[i] = ov::Dimension(std::min(arg_dim.get_min_length(), target_dim.get_min_length()),
                                                std::max(arg_dim.get_max_length(), target_dim.get_max_length()));
            }
            continue;
        }

        const auto& arg_shape_dim = arg_shape[i].get_length();
        const auto& target_shape_dim = target_shape[i].get_length();
        NODE_VALIDATION_CHECK(this_ptr,
                              arg_shape_dim == 1 || target_shape[i] == 1 || arg_shape_dim == target_shape_dim,
                              "Broadcast incorrect target shape. Expecting either 1 or ",
                              arg_shape_dim,
                              ". Got ",
                              target_shape[i]);
        result_shape[i] = std::max(arg_shape_dim, target_shape_dim);
    }
    return result_shape;
}
}  // namespace

bool ov::op::v3::Broadcast::broadcast_evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (get_broadcast_spec().m_type == op::BroadcastType::BIDIRECTIONAL) {
        const auto& arg_shape = inputs[0].get_shape();
        ov::Shape target_shape = op::util::BroadcastBase::get_target_shape(inputs[1]);
        ov::PartialShape result_shape =
            get_result_shape_bidirectional(this, ov::PartialShape{arg_shape}, ov::PartialShape{target_shape});
        auto pair_broadcast_axes = get_broadcast_axes_bidirectional(arg_shape, result_shape.to_shape());
        return op::util::BroadcastBase::evaluate_broadcast(inputs[0],
                                                           outputs[0],
                                                           pair_broadcast_axes,
                                                           result_shape.to_shape());
    }
    return op::util::BroadcastBase::evaluate(outputs, inputs);
}

void ov::op::v3::Broadcast::validate_and_infer_types() {
    OV_OP_SCOPE(v3_Broadcast_validate_and_infer_types);
    if (m_mode.m_type == BroadcastType::NONE) {
        NODE_VALIDATION_CHECK(this,
                              get_input_size() == 3,
                              "axes_mapping input should be provided if explicit mode is used");
    } else {
        NODE_VALIDATION_CHECK(this,
                              get_input_size() == 2,
                              "axes_mapping input should not be provided for mode other than explicit");
    }

    const auto& shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.is_integral_number(),
                          "Broadcast shape must be an integral number, but is: ",
                          shape_et);
    if (m_mode.m_type == BroadcastType::NONE) {
        // axes_mapping node should have integer data type. For now we only allow i64
        const auto& axes_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              axes_et.is_integral_number(),
                              "Broadcast axes must be integral numbers, but are: ",
                              axes_et);
    }

    std::vector<ov::PartialShape> input_shapes;
    const auto& arg_shape = get_input_partial_shape(0);
    const auto& target_shape = get_input_partial_shape(1);
    if (input_values().size() == 2) {
        input_shapes = {arg_shape, target_shape};
    } else {
        const auto& axes_mapping = get_input_partial_shape(2);
        input_shapes = {arg_shape, target_shape, axes_mapping};
    }

    const auto output_shapes = shape_infer(this, input_shapes);

    set_input_is_relevant_to_shape(0);  // arg - Result element type
    set_input_is_relevant_to_shape(1);  // target_shape - Result shape
    if (get_input_size() == 3) {
        set_input_is_relevant_to_shape(2);  // axes_mapping - Broadcast type
    }
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v3::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_Broadcast_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<v3::Broadcast>(new_args.at(0), new_args.at(1), m_mode);
    } else if (new_args.size() == 3) {
        return std::make_shared<v3::Broadcast>(new_args.at(0), new_args.at(1), new_args.at(2), m_mode);
    } else {
        OPENVINO_THROW("Not supported number of Broadcast:v3 args");
    }
}

bool ov::op::v3::Broadcast::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_Broadcast_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

bool ov::op::v3::Broadcast::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v3_Broadcast_evaluate);
    return broadcast_evaluate(outputs, inputs);
}

bool ov::op::v3::Broadcast::has_evaluate() const {
    OV_OP_SCOPE(v3_Broadcast_has_evaluate);
    return m_mode.m_type == BroadcastType::NONE || m_mode.m_type == BroadcastType::PDPD ||
           m_mode.m_type == BroadcastType::NUMPY || m_mode.m_type == BroadcastType::BIDIRECTIONAL;
}

namespace {
using namespace ov::op;
BroadcastModeSpec to_broadcast_mode(const AutoBroadcastSpec& bs) {
    BroadcastModeSpec broadcast_mode;
    broadcast_mode.m_axis = bs.m_axis;
    switch (bs.m_type) {
    case AutoBroadcastType::NONE:
        broadcast_mode.m_type = BroadcastType::NONE;
        break;
    case AutoBroadcastType::NUMPY:
        broadcast_mode.m_type = BroadcastType::NUMPY;
        break;
    case AutoBroadcastType::PDPD:
        broadcast_mode.m_type = BroadcastType::PDPD;
        break;
    }
    return broadcast_mode;
}
}  // namespace

ov::op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                                 const Output<Node>& target_shape,
                                 const Output<Node>& axes_mapping,
                                 const AutoBroadcastSpec& broadcast_spec)
    : util::BroadcastBase{arg, target_shape, axes_mapping, to_broadcast_mode(broadcast_spec)},
      m_broadcast_spec{broadcast_spec} {
    constructor_validate_and_infer_types();
}

ov::op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                                 const Output<Node>& target_shape,
                                 const AutoBroadcastSpec& broadcast_spec)
    : util::BroadcastBase{arg,
                          target_shape,
                          op::v0::Constant::create(element::u8, ov::Shape{}, {0})->output(0),
                          to_broadcast_mode(broadcast_spec)},
      m_broadcast_spec{broadcast_spec} {
    constructor_validate_and_infer_types();
}

void ov::op::v1::Broadcast::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Broadcast_validate_and_infer_types);
    // m_type is deduced and not always explicitly stated, for cases where broadcast
    // has 2 inputs its always NUMPY mode
    if (m_broadcast_spec.m_type == AutoBroadcastType::NONE && get_input_size() < 3) {
        m_broadcast_spec.m_type = AutoBroadcastType::NUMPY;
    }

    // Mocking axes_mapping input for cases that don't require it
    if (m_broadcast_spec.m_type == AutoBroadcastType::NUMPY && get_input_size() < 3) {
        auto output = op::v0::Constant::create(element::u8, ov::Shape{}, {0})->output(0);
        set_argument(2, output);
    }

    // update the base class' mode spec
    auto base_spec = to_broadcast_mode(m_broadcast_spec);
    if (util::BroadcastBase::m_mode.m_type != base_spec.m_type) {
        util::BroadcastBase::m_mode = base_spec;
    }

    const auto& shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.is_integral_number(),
                          "Broadcast shape must be an integral number, but is: ",
                          shape_et);
    if (m_mode.m_type == BroadcastType::NONE) {
        // axes_mapping node should have integer data type. For now we only allow i64
        const auto& axes_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              axes_et.is_integral_number(),
                              "Broadcast axes must be integral numbers, but are: ",
                              axes_et);
    }

    const auto& arg_shape = get_input_partial_shape(0);
    const auto& target_shape = get_input_partial_shape(1);
    const auto& axes_mapping = get_input_partial_shape(2);

    std::vector<ov::PartialShape> input_shapes = {arg_shape, target_shape, axes_mapping};
    const auto output_shapes = shape_infer(this, input_shapes);

    set_input_is_relevant_to_shape(0);  // arg - Result element type
    set_input_is_relevant_to_shape(1);  // target_shape - Result shape
    set_input_is_relevant_to_shape(2);  // axes_mapping - Broadcast type
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v1::Broadcast::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Broadcast_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::Broadcast>(new_args.at(0), new_args.at(1), new_args.at(2), m_broadcast_spec);
}

bool ov::op::v1::Broadcast::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Broadcast_visit_attributes);
    visitor.on_attribute("mode", m_broadcast_spec);
    return true;
}

bool ov::op::v1::Broadcast::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Broadcast_evaluate);
    return op::util::BroadcastBase::evaluate(outputs, inputs);
}

bool ov::op::v1::Broadcast::has_evaluate() const {
    OV_OP_SCOPE(v1_Broadcast_has_evaluate);
    return m_mode.m_type == BroadcastType::NONE || m_mode.m_type == BroadcastType::PDPD ||
           m_mode.m_type == BroadcastType::NUMPY;
}
