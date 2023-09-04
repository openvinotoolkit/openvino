// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/opt_kernel/reshape.hpp"

#include <algorithm>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/reshape.hpp"
#include "reshape_shape_inference.hpp"

using namespace std;

namespace ov {
namespace op {
OPENVINO_SUPPRESS_DEPRECATED_START
namespace reshape {
namespace {
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const AxisVector& order) {
    ngraph::runtime::opt_kernel::reshape(arg0->get_data_ptr<char>(),
                                         out->get_data_ptr<char>(),
                                         arg0->get_shape(),
                                         order,
                                         out->get_shape(),
                                         arg0->get_element_type().size());
    return true;
}
}  // namespace
}  // namespace reshape

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

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto input_shapes = ov::get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

shared_ptr<Node> Reshape::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Reshape_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Reshape>(new_args.at(0), new_args.at(1), m_special_zero);
}

bool Reshape::evaluate_reshape(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    std::vector<PartialShape> input_shapes;
    input_shapes.reserve(inputs.size());
    for (const auto& in : inputs) {
        input_shapes.push_back(in->get_partial_shape());
    }

    auto output_shapes = shape_infer(this, input_shapes, make_tensor_accessor(inputs));
    outputs[0]->set_shape(output_shapes[0].to_shape());

    OPENVINO_SUPPRESS_DEPRECATED_START
    const AxisVector order = ngraph::get_default_order(inputs[0]->get_shape());
    OPENVINO_SUPPRESS_DEPRECATED_END
    return ov::op::reshape::evaluate(inputs[0], outputs[0], order);
}

bool Reshape::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Reshape_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 2));
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
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

bool Reshape::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool Reshape::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    if (get_output_partial_shape(0).is_dynamic() || is_const_fold_disabled()) {
        return false;
    }

    const auto& shape = get_output_shape(0);

    if (auto data_const = std::dynamic_pointer_cast<op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        output_values[0] = std::make_shared<op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
