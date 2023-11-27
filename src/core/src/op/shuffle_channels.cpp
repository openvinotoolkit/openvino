// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/shuffle_channels.hpp"

#include <numeric>
#include <shuffle_channels_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/shuffle_channels.hpp"

using namespace std;
using namespace ngraph;

op::ShuffleChannels::ShuffleChannels(const Output<Node>& data, const int64_t axis, const int64_t group)
    : Op({data}),
      m_axis(axis),
      m_group{group} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ShuffleChannels::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ShuffleChannels_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("group", m_group);
    return true;
}

size_t op::ShuffleChannels::get_zero_based_axis() const {
    const auto input_rank = get_input_partial_shape(0).rank();
    if (input_rank.is_static()) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        return ov::normalize_axis(this, m_axis, input_rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
    } else {
        OPENVINO_THROW("Cannot request zero-based axis with a input of unknown rank");
    }
}

void op::ShuffleChannels::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);

    const auto output_shape = shape_infer(this, ov::util::get_node_input_partial_shapes(*this)).front();
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

OPENVINO_SUPPRESS_DEPRECATED_START
bool op::ShuffleChannels::evaluate_shuffle_channels(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const {
    const auto arg = inputs[0]->get_data_ptr<const char>();
    auto out = outputs[0]->get_data_ptr<char>();
    const auto data_shape = inputs[0]->get_shape();
    const size_t elem_size = inputs[0]->get_element_type().size();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(data_shape);

    ov::reference::shuffle_channels(arg, out, data_shape, elem_size, m_axis, m_group);

    return true;
}
bool op::ShuffleChannels::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_ShuffleChannels_evaluate);
    return evaluate_shuffle_channels(outputs, inputs);
}
OPENVINO_SUPPRESS_DEPRECATED_END

bool op::ShuffleChannels::has_evaluate() const {
    OV_OP_SCOPE(v0_ShuffleChannels_has_evaluate);
    return true;
}

void op::v0::ShuffleChannels::set_axis(int64_t axis) {
    m_axis = axis;
}

void op::v0::ShuffleChannels::set_group(int64_t group) {
    m_group = group;
}
