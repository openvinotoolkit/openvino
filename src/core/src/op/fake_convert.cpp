// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "fake_convert_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
namespace op {
namespace v13 {
namespace fake_convert {
static const std::vector<std::string>& get_valid_types() {
    static const std::vector<std::string> valid_types{"f8e4m3", "f8e5m2"};
    return valid_types;
}
}  // namespace fake_convert

FakeConvert::FakeConvert(const ov::Output<ov::Node>& arg,
                         const ov::Output<ov::Node>& scale,
                         std::string destination_type)
    : Op({arg, scale}),
      m_destination_type(std::move(destination_type)) {
    constructor_validate_and_infer_types();
}

FakeConvert::FakeConvert(const ov::Output<ov::Node>& arg,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& shift,
                         std::string destination_type)
    : Op({arg, scale, shift}),
      m_destination_type(std::move(destination_type)) {
    constructor_validate_and_infer_types();
}

const std::string& FakeConvert::get_destination_type() const {
    return m_destination_type;
}

void FakeConvert::validate_and_infer_types() {
    OV_OP_SCOPE(v13_FakeConvert_validate_and_infer_types);
    validate_destination_type();
    auto out_type = ov::element::Type(element::dynamic);
    for (size_t i = 0; i < get_input_size(); i++) {
        OPENVINO_ASSERT(element::Type::merge(out_type, out_type, get_input_element_type(i)),
                        "Mixed input types are not supported.");
    }
    switch (out_type) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::dynamic:
        break;
    default:
        OPENVINO_THROW("The element type of the input tensor must be a bf16, f16, f32 or dynamic (got ",
                       out_type,
                       ").");
    }
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, out_type, output_shapes[0]);
}

std::shared_ptr<ov::Node> FakeConvert::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v13_FakeConvert_clone_with_new_inputs);
    if (new_args.size() == 2) {
        return std::make_shared<FakeConvert>(new_args.at(0), new_args.at(1), m_destination_type);
    } else if (new_args.size() == 3) {
        return std::make_shared<FakeConvert>(new_args.at(0), new_args.at(1), new_args.at(2), m_destination_type);
    } else {
        OPENVINO_THROW("Incorrect number of FakeConvert new arguments.");
    }
}

bool FakeConvert::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_FakeConvert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

void FakeConvert::validate_destination_type() const {
    const auto& valid_types = fake_convert::get_valid_types();
    OPENVINO_ASSERT(std::find(valid_types.begin(), valid_types.end(), m_destination_type) != valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

bool FakeConvert::has_evaluate() const {
    return false;
}

}  // namespace v13
}  // namespace op
}  // namespace ov
