// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include <numeric>

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"  // tbr
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/reference/shuffle_channels.hpp"
#include "shuffle_channels_shape_inference.hpp"
#include "validation_util.hpp"

using namespace ngraph;

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
        return ov::util::normalize(m_axis, input_rank.get_length());
    } else {
        OPENVINO_THROW("Cannot request zero-based axis with a input of unknown rank");
    }
}

void ShuffleChannels::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, get_input_element_type(0), output_shape);
}

std::shared_ptr<Node> ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

OPENVINO_SUPPRESS_DEPRECATED_START
bool ShuffleChannels::evaluate_shuffle_channels(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const auto arg = inputs[0]->get_data_ptr<const char>();
    auto out = outputs[0]->get_data_ptr<char>();
    const auto data_shape = inputs[0]->get_shape();
    const size_t elem_size = inputs[0]->get_element_type().size();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(data_shape);

    reference::shuffle_channels(arg, out, data_shape, elem_size, m_axis, m_group);

    return true;
}
bool ShuffleChannels::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_ShuffleChannels_evaluate);
    return evaluate_shuffle_channels(outputs, inputs);
}
OPENVINO_SUPPRESS_DEPRECATED_END

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
