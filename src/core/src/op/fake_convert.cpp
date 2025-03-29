// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include "element_visitor.hpp"
#include "fake_convert_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/fake_convert.hpp"

namespace ov {
namespace op {
namespace v13 {
namespace fake_convert_details {
static const std::vector<ov::element::Type>& get_valid_types() {
    static const std::vector<ov::element::Type> valid_types{ov::element::f8e4m3, ov::element::f8e5m2};
    return valid_types;
}

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;
    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(ov::TensorVector& outputs,
                             const ov::TensorVector& inputs,
                             const ov::element::Type& destination_type) {
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

FakeConvert::FakeConvert(const ov::Output<ov::Node>& data,
                         const ov::Output<ov::Node>& scale,
                         std::string destination_type)
    : FakeConvert(data, scale, ov::element::Type(destination_type)) {}

FakeConvert::FakeConvert(const ov::Output<ov::Node>& data,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& shift,
                         std::string destination_type)
    : FakeConvert(data, scale, shift, ov::element::Type(destination_type)) {}

FakeConvert::FakeConvert(const ov::Output<ov::Node>& data,
                         const ov::Output<ov::Node>& scale,
                         const ov::element::Type& destination_type)
    : Op({data, scale}),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

FakeConvert::FakeConvert(const ov::Output<ov::Node>& data,
                         const ov::Output<ov::Node>& scale,
                         const ov::Output<ov::Node>& shift,
                         const ov::element::Type& destination_type)
    : Op({data, scale, shift}),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

std::string FakeConvert::get_destination_type() const {
    return m_destination_type.get_type_name();
}

void FakeConvert::set_destination_type(ov::element::Type destination_type) {
    m_destination_type = destination_type;
}

const ov::element::Type& FakeConvert::get_destination_element_type() const {
    return m_destination_type;
}

void FakeConvert::validate_and_infer_types() {
    OV_OP_SCOPE(v13_FakeConvert_validate_and_infer_types);
    validate_destination_type();
    auto out_type = get_input_element_type(0);
    for (size_t i = 1; i < get_input_size(); i++) {
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
        OPENVINO_THROW("The element type of the input tensor must be a bf16, f16, f32 but got: ", out_type);
    }
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
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
    const auto& valid_types = fake_convert_details::get_valid_types();
    const auto is_supported_type =
        std::find(valid_types.begin(), valid_types.end(), m_destination_type) != valid_types.end();
    OPENVINO_ASSERT(is_supported_type, "Bad format for f8 conversion type: ", m_destination_type);
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
    return IF_TYPE_OF(v13_FakeConvert_evaluate,
                      OV_PP_ET_LIST(bf16, f16, f32),
                      fake_convert_details::Evaluate,
                      inputs[0].get_element_type(),
                      outputs,
                      inputs,
                      get_destination_element_type());

    return true;
}
}  // namespace v13
}  // namespace op
}  // namespace ov
