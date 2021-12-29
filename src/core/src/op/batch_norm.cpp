// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/batch_norm.hpp"

#include <sstream>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v1::BatchNormInference);

ov::op::v1::BatchNormInference::BatchNormInference(const Output<Node>& input,
                                                   const Output<Node>& gamma,
                                                   const Output<Node>& beta,
                                                   const Output<Node>& mean,
                                                   const Output<Node>& variance,
                                                   double epsilon)
    : Op({gamma, beta, input, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool ov::op::v1::BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void ov::op::v1::BatchNormInference::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_BatchNormInference_validate_and_infer_types);
    element::Type result_et;
    ov::PartialShape result_batch_shape;
    ov::PartialShape result_channel_shape;  // unused here

    NODE_VALIDATION_CHECK(this,
                          m_epsilon >= 0,
                          "Attribute 'epsilon' must be a floating-point value greater than or equal to zero. Got: ",
                          m_epsilon);

    set_output_size(1);
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

    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<ov::Node> ov::op::v1::BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(2),
                                                new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}

BWDCMP_RTTI_DEFINITION(ov::op::v5::BatchNormInference);

ov::op::v5::BatchNormInference::BatchNormInference(const Output<Node>& input,
                                                   const Output<Node>& gamma,
                                                   const Output<Node>& beta,
                                                   const Output<Node>& mean,
                                                   const Output<Node>& variance,
                                                   double epsilon)
    : Op({input, gamma, beta, mean, variance}),
      m_epsilon(epsilon) {
    constructor_validate_and_infer_types();
}

bool ov::op::v5::BatchNormInference::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v5_BatchNormInference_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void ov::op::v5::BatchNormInference::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v5_BatchNormInference_validate_and_infer_types);
    element::Type result_et;
    ov::PartialShape result_batch_shape;
    ov::PartialShape result_channel_shape;  // unused here

    NODE_VALIDATION_CHECK(this,
                          m_epsilon >= 0,
                          "Attribute 'epsilon' must be a floating-point value greater than or equal to zero. Got: ",
                          m_epsilon);

    set_output_size(1);
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

    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<ov::Node> ov::op::v5::BatchNormInference::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v5_BatchNormInference_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                new_args.at(4),
                                                m_epsilon);
}
