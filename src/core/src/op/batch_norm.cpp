// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <sstream>

#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {

op::v0::BatchNormInference::BatchNormInference(const Output<Node>& input,
                                               const Output<Node>& gamma,
                                               const Output<Node>& beta,
                                               const Output<Node>& mean,
                                               const Output<Node>& variance,
                                               double epsilon)
    : Op({gamma, beta, input, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool op::v0::BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void op::v0::BatchNormInference::validate_and_infer_types() {
    OV_OP_SCOPE(v0_BatchNormInference_validate_and_infer_types);
    element::Type result_et;
    ov::PartialShape result_batch_shape;
    ov::PartialShape result_channel_shape;  // unused here

    NODE_VALIDATION_CHECK(this,
                          m_epsilon >= 0,
                          "Attribute 'epsilon' must be a floating-point value greater than or equal to zero. Got: ",
                          m_epsilon);

    set_output_size(1);
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        ngraph::infer_batch_norm_forward(this,
                                         get_input_element_type(INPUT_DATA),
                                         get_input_element_type(INPUT_GAMMA),
                                         get_input_element_type(INPUT_BETA),
                                         get_input_element_type(INPUT_MEAN),
                                         get_input_element_type(INPUT_VARIANCE),
                                         get_input_partial_shape(INPUT_DATA),
                                         get_input_partial_shape(INPUT_GAMMA),
                                         get_input_partial_shape(INPUT_BETA),
                                         get_input_partial_shape(INPUT_MEAN),
                                         get_input_partial_shape(INPUT_VARIANCE));
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<Node> op::v0::BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(2),
                                                new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}

op::v5::BatchNormInference::BatchNormInference(const Output<Node>& input,
                                               const Output<Node>& gamma,
                                               const Output<Node>& beta,
                                               const Output<Node>& mean,
                                               const Output<Node>& variance,
                                               double epsilon)
    : Op({input, gamma, beta, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool op::v5::BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void op::v5::BatchNormInference::validate_and_infer_types() {
    OV_OP_SCOPE(v5_BatchNormInference_validate_and_infer_types);
    element::Type result_et;
    ov::PartialShape result_batch_shape;
    ov::PartialShape result_channel_shape;  // unused here

    NODE_VALIDATION_CHECK(this,
                          m_epsilon >= 0,
                          "Attribute 'epsilon' must be a floating-point value greater than or equal to zero. Got: ",
                          m_epsilon);

    set_output_size(1);
    OPENVINO_SUPPRESS_DEPRECATED_START
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        ngraph::infer_batch_norm_forward(this,
                                         get_input_element_type(INPUT_DATA),
                                         get_input_element_type(INPUT_GAMMA),
                                         get_input_element_type(INPUT_BETA),
                                         get_input_element_type(INPUT_MEAN),
                                         get_input_element_type(INPUT_VARIANCE),
                                         get_input_partial_shape(INPUT_DATA),
                                         get_input_partial_shape(INPUT_GAMMA),
                                         get_input_partial_shape(INPUT_BETA),
                                         get_input_partial_shape(INPUT_MEAN),
                                         get_input_partial_shape(INPUT_VARIANCE));
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<Node> op::v5::BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}
}  // namespace ov
