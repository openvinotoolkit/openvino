// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_normalization.hpp"

#include "group_normalization_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
op::v12::GroupNormalization::GroupNormalization() : m_num_groups{0}, m_epsilon{0} {}

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
    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, get_input_element_type(0), output_shapes.at(0));
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
}  // namespace ov
