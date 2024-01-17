// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_norm.hpp"

#include <sstream>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "validation_util.hpp"

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

namespace {
struct ChannelShapedInputSpec {
    const element::Type& m_element_type;
    const PartialShape& m_shape;
    std::string m_input_name;
};

std::tuple<element::Type, PartialShape, PartialShape> infer_batch_norm_forward(
    const Node* node,
    const element::Type& input_element_type,
    const element::Type& gamma_element_type,
    const element::Type& beta_element_type,
    const element::Type& mean_element_type,
    const element::Type& variance_element_type,
    const PartialShape& input_shape,
    const PartialShape& gamma_shape,
    const PartialShape& beta_shape,
    const PartialShape& mean_shape,
    const PartialShape& variance_shape) {
    const std::vector<ChannelShapedInputSpec> channel_shaped_inputs{
        {gamma_element_type, gamma_shape, "gamma"},
        {beta_element_type, beta_shape, "beta"},
        {mean_element_type, mean_shape, "mean"},
        {variance_element_type, variance_shape, "variance"}};

    std::stringstream ss;
    bool first = true;
    for (const auto& inp : channel_shaped_inputs) {
        if (!first) {
            ss << "/";
        }
        ss << inp.m_input_name;
        first = false;
    }
    // slash separated string naming all the channel-shaped inputs, for use in error messages.
    const auto channel_input_names = ss.str();

    // Infer output element type.
    element::Type et_result{input_element_type};

    for (const auto& inp : channel_shaped_inputs) {
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et_result, et_result, inp.m_element_type),
                              "Input element types do not match.");
    }

    NODE_VALIDATION_CHECK(node,
                          et_result.is_dynamic() || et_result.is_real(),
                          "Input element types must be floating-point. Got: ",
                          et_result);

    // Extract channel dimension from input shape.
    Dimension channel_dim{Dimension::dynamic()};

    const auto input_rank = input_shape.rank();
    if (input_rank.is_static()) {
        NODE_VALIDATION_CHECK(node,
                              input_rank.get_length() >= 2,
                              "Input argument must have rank of at least 2 (input argument shape: ",
                              input_shape,
                              ").");

        channel_dim = input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size "channel_dim".
    PartialShape channel_shape{PartialShape::dynamic()};

    for (const auto& inp : channel_shaped_inputs) {
        NODE_VALIDATION_CHECK(node,
                              PartialShape::merge_into(channel_shape, inp.m_shape),
                              "Shapes for ",
                              channel_input_names,
                              " do not match.");
    }

    NODE_VALIDATION_CHECK(node,
                          channel_shape.merge_rank(1),
                          "Shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(node,
                          Dimension::merge(channel_dim, channel_dim, channel_shape[0]),
                          "Input channel dimension (",
                          channel_dim,
                          ") does not match shape for ",
                          channel_input_names,
                          " (",
                          channel_shape,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          channel_dim.is_dynamic() || channel_dim.get_length() >= 1,
                          "Channel count must be at least 1.");

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    PartialShape batch_result_shape{input_shape};

    if (batch_result_shape.rank().is_static()) {
        batch_result_shape[1] = channel_dim;
    }

    return std::make_tuple(et_result, batch_result_shape, PartialShape{channel_dim});
}
}  // namespace

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
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
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
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
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
