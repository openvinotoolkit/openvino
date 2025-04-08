// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include <cstring>

#include "itt.hpp"
#include "squeeze_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
Squeeze::Squeeze() : util::SqueezeBase() {}

Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes) : util::SqueezeBase(data, axes) {
    constructor_validate_and_infer_types();
}

Squeeze::Squeeze(const Output<Node>& data) : util::SqueezeBase(data) {
    constructor_validate_and_infer_types();
}

void Squeeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Squeeze_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> Squeeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Squeeze_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    switch (new_args.size()) {
    case 1:
        return std::make_shared<Squeeze>(new_args[0]);
    case 2:
        return std::make_shared<Squeeze>(new_args[0], new_args[1]);
    default:
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

bool Squeeze::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto output_shapes =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
    outputs[0].set_shape(output_shapes.front().get_shape());

    std::memcpy(outputs[0].data(), inputs[0].data(), outputs[0].get_byte_size());
    return true;
}

}  // namespace v0

namespace v15 {
Squeeze::Squeeze() : util::SqueezeBase() {}

Squeeze::Squeeze(const Output<Node>& data, const bool allow_axis_skip)
    : util::SqueezeBase(data),
      m_allow_axis_skip{allow_axis_skip} {
    constructor_validate_and_infer_types();
}

Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes, const bool allow_axis_skip)
    : util::SqueezeBase(data, axes),
      m_allow_axis_skip{allow_axis_skip} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Squeeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_Squeeze_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    switch (new_args.size()) {
    case 1:
        return std::make_shared<Squeeze>(new_args[0], m_allow_axis_skip);
    case 2:
        return std::make_shared<Squeeze>(new_args[0], new_args[1], m_allow_axis_skip);
    default:
        OPENVINO_THROW("Incorrect number of new arguments");
    }
}

void Squeeze::validate_and_infer_types() {
    OV_OP_SCOPE(v15_Squeeze_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool Squeeze::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_Squeeze_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto output_shapes =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs));
    outputs[0].set_shape(output_shapes.front().get_shape());

    std::memcpy(outputs[0].data(), inputs[0].data(), outputs[0].get_byte_size());
    return true;
}

bool Squeeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_Squeeze_visit_attributes);
    visitor.on_attribute("allow_axis_skip", m_allow_axis_skip);
    return true;
}

bool Squeeze::get_allow_axis_skip() const {
    OV_OP_SCOPE(v15_Squeeze_get_allow_axis_skip);
    return m_allow_axis_skip;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
