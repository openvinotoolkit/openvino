// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"

namespace ov {
namespace op {
namespace v13 {
namespace fake_convert {
static const std::vector<std::string>& get_valid_types() {
    static const std::vector<std::string> valid_types{"f8e4m3", "f8e5m2"};
    return valid_types;
}

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;
};

}  // namespace fake_convert
FakeConvert::FakeConvert(const ov::Output<ov::Node>& arg,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& shift,
                         std::string destination_type,
                         bool apply_scale)
    : Op({arg, scale, shift}),
      m_destination_type(std::move(destination_type)),
      m_apply_scale(apply_scale) {
    constructor_validate_and_infer_types();
}

bool FakeConvert::get_apply_scale() const {
    return m_apply_scale;
}

const std::string& FakeConvert::get_destination_type() const {
    return m_destination_type;
}

void FakeConvert::validate_and_infer_types() {
    OV_OP_SCOPE(v13_FakeConvert_validate_and_infer_types);
    validate_type();
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> FakeConvert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v13_FakeConvert_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 3, "Incorrect number of new arguments");

    return std::make_shared<FakeConvert>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_destination_type,
                                         m_apply_scale);
}

bool FakeConvert::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_FakeConvert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    visitor.on_attribute("apply_scale", m_apply_scale);

    return true;
}

void FakeConvert::validate_type() const {
    const auto& valid_types = fake_convert::get_valid_types();
    OPENVINO_ASSERT(std::find(valid_types.begin(), valid_types.end(), m_destination_type) != valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

bool FakeConvert::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v13_FakeConvert_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 3);

    const auto& data_shape = inputs[0].get_shape();
    outputs[0].set_shape(data_shape);

    return true;
}

bool FakeConvert::has_evaluate() const {
    OV_OP_SCOPE(v13_FakeConvert_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

}  // namespace v13
}  // namespace op
}  // namespace ov
