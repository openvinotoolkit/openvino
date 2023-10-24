// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unsqueeze.hpp"

#include <cstddef>
#include <functional>
#include <set>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "unsqueeze_shape_inference.hpp"

ov::op::v0::Unsqueeze::Unsqueeze(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& axes)
    : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

void ov::op::v0::Unsqueeze::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Unsqueeze_validate_and_infer_types);

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

bool ov::op::v0::Unsqueeze::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Unsqueeze_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v0::Unsqueeze::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Unsqueeze_clone_with_new_inputs);
    if (new_args.size() != 2) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return std::make_shared<Unsqueeze>(new_args.at(0), new_args.at(1));
}

bool ov::op::v0::Unsqueeze::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Unsqueeze_evaluate);
    OPENVINO_ASSERT(inputs.size() == 2);
    if (outputs.empty()) {
        outputs.emplace_back(ov::Tensor(inputs[0].get_element_type(), {0}));
    } else {
        OPENVINO_ASSERT(outputs.size() == 1);
    }
    const auto& output_shape = shape_infer(this,
                                           std::vector<ov::PartialShape>{inputs[0].get_shape(), inputs[1].get_shape()},
                                           make_tensor_accessor(inputs))
                                   .front()
                                   .to_shape();
    outputs[0].set_shape(output_shape);
    std::memcpy(outputs[0].data(), inputs[0].data(), outputs[0].get_byte_size());
    return true;
}

bool ov::op::v0::Unsqueeze::has_evaluate() const {
    OV_OP_SCOPE(v0_Unsqueeze_has_evaluate);
    return true;
}

bool ov::op::v0::Unsqueeze::evaluate_lower(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ov::op::v0::Unsqueeze::evaluate_upper(ov::TensorVector& output_values) const {
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ov::op::v0::Unsqueeze::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool ov::op::v0::Unsqueeze::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto shape = get_output_partial_shape(0).to_shape();

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}
