// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"

using namespace ov;

op::v12::GroupNormalization::GroupNormalization(const Output<Node>& data,
                                                const Output<Node>& scale,
                                                const Output<Node>& bias,
                                                int64_t num_groups,
                                                double epsilon)
    : Op({data, scale, bias}),
      m_num_groups{num_groups},
      m_epsilon{epsilon} {
    constructor_validate_and_infer_types();
}

bool op::v12::GroupNormalization::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v12_GroupNormalization_visit_attributes);
    visitor.on_attribute("num_groups", m_num_groups);
    visitor.on_attribute("epsilon", m_epsilon);
    return true;
}

void op::v12::GroupNormalization::validate_and_infer_types() {
    OV_OP_SCOPE(v12_GroupNormalization_validate_and_infer_types);
    const auto data_partial_shape = get_input_partial_shape(0);
    const auto data_rank = data_partial_shape.rank();
    const auto scale_partial_shape = get_input_partial_shape(1);
    const auto bias_partial_shape = get_input_partial_shape(2);

    NODE_VALIDATION_CHECK(this, get_num_groups() > 0, "The number of groups needs to be a positive integer value");

    NODE_VALIDATION_CHECK(this,
                          scale_partial_shape.rank().compatible(Dimension{1}),
                          "The scale input is required to be 1D");
    NODE_VALIDATION_CHECK(this,
                          bias_partial_shape.rank().compatible(Dimension{1}),
                          "The bias input is required to be 1D");

    NODE_VALIDATION_CHECK(this,
                          data_rank.is_dynamic() || data_rank.get_length() >= 2,
                          "The input tensor is required to be at least 2D");

    if (data_rank.is_static()) {
        const auto channels_dim = data_partial_shape[1];
        NODE_VALIDATION_CHECK(
            this,
            scale_partial_shape.rank().is_dynamic() || channels_dim.compatible(scale_partial_shape[0]),
            "The scale input shape needs to match the channel dimension in the data input");
        NODE_VALIDATION_CHECK(this,
                              bias_partial_shape.rank().is_dynamic() || channels_dim.compatible(bias_partial_shape[0]),
                              "The bias input shape needs to match the channel dimension in the data input");

        NODE_VALIDATION_CHECK(this,
                              channels_dim.is_dynamic() || get_num_groups() <= channels_dim.get_length(),
                              "The number of groups must not exceed the number of channels in the input tensor");

        NODE_VALIDATION_CHECK(this,
                              channels_dim.is_dynamic() || channels_dim.get_length() % get_num_groups() == 0,
                              "The number of channels is required to be evenly divisible by the number of groups");
    }

    NODE_VALIDATION_CHECK(this,
                          scale_partial_shape.compatible(bias_partial_shape),
                          "The shapes of both scale and bias inputs need to match");

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v12::GroupNormalization::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v12_GroupNormalization_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GroupNormalization>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_num_groups,
                                                m_epsilon);
}
