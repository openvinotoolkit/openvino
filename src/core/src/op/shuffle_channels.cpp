// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include <numeric>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/shuffle_channels.hpp"
#include "shuffle_channels_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
ShuffleChannels::ShuffleChannels(const Output<Node>& data, const int64_t axis, const int64_t group)
    : Op({data}),
      m_axis(axis),
      m_group{group} {
    constructor_validate_and_infer_types();
}

bool ShuffleChannels::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ShuffleChannels_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("group", m_group);
    return true;
}

size_t ShuffleChannels::get_zero_based_axis() const {
    const auto input_rank = get_input_partial_shape(0).rank();
    if (input_rank.is_static()) {
        return ov::util::try_normalize_axis(m_axis, input_rank, *this);
    } else {
        OPENVINO_THROW("Cannot request zero-based axis with a input of unknown rank");
    }
}

void ShuffleChannels::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

bool ShuffleChannels::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_ShuffleChannels_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& input = inputs[0];
    const auto& data_shape = input.get_shape();

    auto& output = outputs[0];
    output.set_shape(data_shape);

    const auto arg = static_cast<const char*>(input.data());
    const auto out = static_cast<char*>(output.data());
    const auto elem_size = input.get_element_type().size();
    reference::shuffle_channels(arg, out, data_shape, elem_size, m_axis, m_group);
    return true;
}

bool ShuffleChannels::has_evaluate() const {
    OV_OP_SCOPE(v0_ShuffleChannels_has_evaluate);
    return true;
}

void ShuffleChannels::set_axis(int64_t axis) {
    m_axis = axis;
}

void ShuffleChannels::set_group(int64_t group) {
    m_group = group;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
