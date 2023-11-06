// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "itt.hpp"

namespace ov {

op::v13::FakeConvert::FakeConvert(const ov::Output<ov::Node>& arg,
                                  const ov::Output<ov::Node>& scale,
                                  const ov::Output<ov::Node>& shift,
                                  const std::string& destination_type,
                                  bool apply_scale)
    : Op({arg, scale, shift}),
      m_destination_type(destination_type),
      m_apply_scale(apply_scale) {
    constructor_validate_and_infer_types();
}

const std::vector<std::string> op::v13::FakeConvert::m_valid_types({"HF8", "BF8"});

void op::v13::FakeConvert::validate_and_infer_types() {
    OV_OP_SCOPE(v13_FakeConvert_validate_and_infer_types);
    validate();
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> op::v13::FakeConvert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v13_FakeConvert_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");

    return std::make_shared<ov::op::v13::FakeConvert>(new_args.at(0),
                                                      new_args.at(1),
                                                      new_args.at(2),
                                                      m_destination_type,
                                                      m_apply_scale);
}

bool op::v13::FakeConvert::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_FakeConvert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    visitor.on_attribute("apply_scale", m_apply_scale);

    return true;
}

void op::v13::FakeConvert::validate() const {
    OPENVINO_ASSERT(std::find(m_valid_types.begin(), m_valid_types.end(), m_destination_type) != m_valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

bool op::v13::FakeConvert::has_evaluate() const {
    return false;
}
}  // namespace ov
