// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <sstream>

#include "batch_norm_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace batch_norm {
namespace {
element::Type infer_output_element_type(const Op* const op,
                                        element::Type data_et,
                                        const std::vector<element::Type>& inputs_element_types) {
    for (const auto& input_et : inputs_element_types) {
        NODE_VALIDATION_CHECK(op,
                              element::Type::merge(data_et, data_et, input_et),
                              "Input element types do not match.");
    }

    NODE_VALIDATION_CHECK(op,
                          data_et.is_dynamic() || data_et.is_real(),
                          "Input element types must be floating-point. Got: ",
                          data_et);

    return data_et;
}
}  // namespace
}  // namespace batch_norm

namespace v0 {

BatchNormInference::BatchNormInference(const Output<Node>& input,
                                       const Output<Node>& gamma,
                                       const Output<Node>& beta,
                                       const Output<Node>& mean,
                                       const Output<Node>& variance,
                                       double epsilon)
    : Op({gamma, beta, input, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void BatchNormInference::validate_and_infer_types() {
    OV_OP_SCOPE(v0_BatchNormInference_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_epsilon >= 0, "Attribute 'epsilon' must be non negative value. Got: ", m_epsilon);

    const auto output_et = batch_norm::infer_output_element_type(this,
                                                                 get_input_element_type(INPUT_DATA),
                                                                 {get_input_element_type(INPUT_GAMMA),
                                                                  get_input_element_type(INPUT_BETA),
                                                                  get_input_element_type(INPUT_MEAN),
                                                                  get_input_element_type(INPUT_VARIANCE)});

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, output_et, output_shapes[0]);
}

std::shared_ptr<Node> BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(2),
                                                new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}
}  // namespace v0

namespace v5 {
BatchNormInference::BatchNormInference(const Output<Node>& input,
                                       const Output<Node>& gamma,
                                       const Output<Node>& beta,
                                       const Output<Node>& mean,
                                       const Output<Node>& variance,
                                       double epsilon)
    : Op({input, gamma, beta, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void BatchNormInference::validate_and_infer_types() {
    OV_OP_SCOPE(v5_BatchNormInference_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, m_epsilon >= 0, "Attribute 'epsilon' must be non negative value. Got: ", m_epsilon);

    const auto output_et = batch_norm::infer_output_element_type(this,
                                                                 get_input_element_type(INPUT_DATA),
                                                                 {get_input_element_type(INPUT_GAMMA),
                                                                  get_input_element_type(INPUT_BETA),
                                                                  get_input_element_type(INPUT_MEAN),
                                                                  get_input_element_type(INPUT_VARIANCE)});

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, output_et, output_shapes[0]);
}

std::shared_ptr<Node> BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}
}  // namespace v5
}  // namespace op
}  // namespace ov
