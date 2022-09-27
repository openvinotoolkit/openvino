// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/shuffle_channels.hpp"

#include <numeric>
#include <shuffle_channels_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/runtime/reference/shuffle_channels.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::ShuffleChannels);

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
    if (m_axis >= 0) {
        return m_axis;
    } else {
        if (!get_input_partial_shape(0).rank().is_dynamic()) {
            return m_axis + get_input_partial_shape(0).rank().get_length();
        } else {
            throw ngraph_error("Cannot request zero-based axis with a input of unknown rank");
        }
    }
}

void op::ShuffleChannels::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ShuffleChannels_validate_and_infer_types);

    const auto& data_type = get_input_element_type(0);
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    const std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

shared_ptr<Node> op::ShuffleChannels::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ShuffleChannels_clone_with_new_inputs);
    if (new_args.size() != 1) {
        throw ngraph_error("Expected 1 element in new_args for the ShuffleChannels op but got " +
                           std::to_string(new_args.size()));
    }

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_group);
}

bool op::ShuffleChannels::evaluate_shuffle_channels(const HostTensorVector& outputs,
                                                    const HostTensorVector& inputs) const {
    const auto arg = inputs[0]->get_data_ptr<const char>();
    auto out = outputs[0]->get_data_ptr<char>();
    const auto data_shape = inputs[0]->get_shape();
    const size_t elem_size = inputs[0]->get_element_type().size();

    outputs[0]->set_element_type(inputs[0]->get_element_type());
    outputs[0]->set_shape(data_shape);

    ngraph::runtime::reference::shuffle_channels(arg, out, data_shape, elem_size, m_axis, m_group);

    return true;
}
bool op::ShuffleChannels::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_ShuffleChannels_evaluate);
    return evaluate_shuffle_channels(outputs, inputs);
}

bool op::ShuffleChannels::has_evaluate() const {
    OV_OP_SCOPE(v0_ShuffleChannels_has_evaluate);
    return true;
}
