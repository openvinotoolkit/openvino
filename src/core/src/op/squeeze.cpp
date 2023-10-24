// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/squeeze.hpp"

#include <cstring>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "squeeze_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
namespace validate {
namespace {

bool axes_has_and_set_bound(const Node& op) {
    return (op.get_input_size() < 2) || op.get_input_tensor(1).has_and_set_bound();
}
}  // namespace
}  // namespace validate

Squeeze::Squeeze() : Op() {}

Squeeze::Squeeze(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

Squeeze::Squeeze(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

void Squeeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Squeeze_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
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

bool Squeeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Squeeze_has_evaluate);
    const auto validate_axes_type = [](const element::Type& et) -> bool {
        switch (et) {
        case element::i8:
        case element::i16:
        case element::i32:
        case element::i64:
        case element::u8:
        case element::u16:
        case element::u32:
        case element::u64:
            return true;
        default:
            return false;
        }
    };

    return (get_input_size() < 2) || validate_axes_type(get_input_element_type(1));
}

bool Squeeze::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_lower);
    return validate::axes_has_and_set_bound(*this) && default_lower_bound_evaluator(this, output_values);
}

bool Squeeze::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v0_Squeeze_evaluate_upper);
    return validate::axes_has_and_set_bound(*this) && default_upper_bound_evaluator(this, output_values);
}

bool Squeeze::evaluate_label(TensorLabelVector& output_labels) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return validate::axes_has_and_set_bound(*this) && default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool Squeeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    OV_OP_SCOPE(v0_Squeeze_constant_fold);
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    if (auto data_const = std::dynamic_pointer_cast<Constant>(inputs_values[0].get_node_shared_ptr())) {
        const auto& shape = get_output_shape(0);
        output_values[0] = std::make_shared<Constant>(*data_const, shape);
        return true;
    }
    return false;
}

bool Squeeze::is_dynamic() const {
    return get_output_partial_shape(0).is_dynamic();
}
}  // namespace v0
}  // namespace op
}  // namespace ov
