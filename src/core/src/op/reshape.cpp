// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <algorithm>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/reshape.hpp"
#include "reshape_shape_inference.hpp"

namespace ov {
namespace op {
namespace v1 {

Reshape::Reshape(const Output<Node>& arg, const Output<Node>& shape_pattern, bool zero_flag)
    : Op({arg, shape_pattern}),
      m_special_zero(zero_flag) {
    ov::mark_as_precision_sensitive(input(1));
    constructor_validate_and_infer_types();
}

bool Reshape::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Reshape_visit_attributes);
    visitor.on_attribute("special_zero", m_special_zero);
    return true;
}
void Reshape::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Reshape_validate_and_infer_types);
    const auto& shape_pattern_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(this,
                          shape_pattern_et.is_integral_number(),
                          "PartialShape pattern must be an integral number.");

    auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

std::shared_ptr<Node> Reshape::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Reshape_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

bool Reshape::evaluate_reshape(TensorVector& outputs, const TensorVector& inputs) const {
    const auto output_shape =
        shape_infer(this, ov::util::get_tensors_partial_shapes(inputs), make_tensor_accessor(inputs))
            .front()
            .to_shape();

    if (outputs.empty()) {
        outputs.emplace_back(inputs[0].get_element_type(), output_shape);
    } else {
        OPENVINO_ASSERT(outputs.size() == 1);
        outputs[0].set_shape(output_shape);
    }

    if (inputs[0].get_element_type() == ov::element::string) {
        ov::reference::reshape(inputs[0].data<std::string>(), outputs[0].data<std::string>(), inputs[0].get_shape());
    } else {
        ov::reference::reshape(static_cast<const char*>(inputs[0].data()),
                               static_cast<char*>(outputs[0].data()),
                               inputs[0].get_shape(),
                               inputs[0].get_element_type().size());
    }
    return true;
}

bool Reshape::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Reshape_evaluate);
    return evaluate_reshape(outputs, inputs);
}

bool Reshape::has_evaluate() const {
    OV_OP_SCOPE(v1_Reshape_has_evaluate);
    const auto& shape_pattern_et = get_input_element_type(1);
    return shape_pattern_et.is_integral_number() && (shape_pattern_et.bitwidth() >= 8);
}

bool Reshape::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Reshape::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Reshape::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    return get_input_tensor(1).has_and_set_bound() && default_symbol_evaluator(this, {0}, output_symbols);
}

bool Reshape::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (!can_constant_fold(inputs_values)) {
        return false;
    }

    if (auto data_const = ov::as_type_ptr<v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<v0::Constant>(*data_const, get_output_shape(0));
        return true;
    } else {
        return false;
    }
}

bool Reshape::can_constant_fold(const OutputVector& input_values) const {
    return get_output_partial_shape(0).is_static() && !is_const_fold_disabled();
}
}  // namespace v1
}  // namespace op
}  // namespace ov
