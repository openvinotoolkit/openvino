// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/op/util/elementwise_args.hpp"

ov::op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob)
    : m_autob(autob) {}

ov::op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                       const Output<Node>& arg1,
                                                                       const AutoBroadcastSpec& autob)
    : Op({arg0, arg1}),
      m_autob(autob) {}

void ov::op::util::BinaryElementwiseArithmetic::validate_and_infer_elementwise_arithmetic() {
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    element::Type& args_et = std::get<0>(args_et_pshape);
    PartialShape& args_pshape = std::get<1>(args_et_pshape);

    const auto is_supported_et = (args_et != element::boolean && args_et != element::string);
    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || is_supported_et,
                          "This operation does not support inputs with element type: ",
                          args_et);

    set_output_type(0, args_et, args_pshape);
}

void ov::op::util::BinaryElementwiseArithmetic::validate_and_infer_types() {
    OV_OP_SCOPE(v0_util_BinaryElementwiseArithmetic_validate_and_infer_types);
    validate_and_infer_elementwise_arithmetic();
}

bool ov::op::util::BinaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_util_BinaryElementwiseArithmetic_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}

bool ov::op::util::BinaryElementwiseArithmetic::evaluate_upper(ov::TensorVector& output_values) const {
    OPENVINO_ASSERT(output_values.size() == 1);
    TensorVector lower_output_tensors;
    for (const auto& output : output_values)
        lower_output_tensors.emplace_back(output.get_element_type(), output.get_shape());

    if (!interval_bound_evaluator(this, lower_output_tensors, output_values))
        return false;
    return true;
}

bool ov::op::util::BinaryElementwiseArithmetic::evaluate_lower(ov::TensorVector& output_values) const {
    OPENVINO_ASSERT(output_values.size() == 1);
    TensorVector upper_output_tensors;
    for (const auto& output : output_values)
        upper_output_tensors.emplace_back(output.get_element_type(), output.get_shape());

    if (!interval_bound_evaluator(this, output_values, upper_output_tensors))
        return false;
    return true;
}
