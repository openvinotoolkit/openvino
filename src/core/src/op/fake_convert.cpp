// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/fake_convert.hpp"

namespace ov {
namespace op {
namespace v13 {
namespace fake_convert_details {
static const std::vector<std::string>& get_valid_types() {
    static const std::vector<std::string> valid_types{"f8e4m3", "f8e5m2"};
    return valid_types;
}

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;
    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(ov::TensorVector& outputs,
                             const ov::TensorVector& inputs,
                             const std::string& destination_type) {
        if (inputs.size() == 2) {  // Default shift
            reference::fake_convert<T>(inputs[0].data<const T>(),
                                       inputs[1].data<const T>(),
                                       outputs[0].data<T>(),
                                       inputs[0].get_shape(),
                                       inputs[1].get_shape(),
                                       destination_type);
        } else {
            reference::fake_convert<T>(inputs[0].data<const T>(),
                                       inputs[1].data<const T>(),
                                       inputs[2].data<const T>(),
                                       outputs[0].data<T>(),
                                       inputs[0].get_shape(),
                                       inputs[1].get_shape(),
                                       inputs[2].get_shape(),
                                       destination_type);
        }
        return true;
    }
};

}  // namespace fake_convert_details
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
    const auto& valid_types = fake_convert_details::get_valid_types();
    OPENVINO_ASSERT(std::find(valid_types.begin(), valid_types.end(), m_destination_type) != valid_types.end(),
                    "Bad format for f8 conversion type: " + m_destination_type);
}

bool FakeConvert::has_evaluate() const {
    OV_OP_SCOPE(v13_FakeConvert_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::bf16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

bool FakeConvert::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v13_FakeConvert_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2 || inputs.size() == 3);

    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IfTypeOf<f16, f32, bf16>::apply<fake_convert_details::Evaluate>(inputs[0].get_element_type(),
                                                                           outputs,
                                                                           inputs,
                                                                           get_destination_type());

    return true;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
