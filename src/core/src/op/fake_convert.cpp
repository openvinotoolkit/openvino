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
namespace fake_convert {
static const std::vector<std::string>& get_valid_types() {
    static const std::vector<std::string> valid_types{"f8e4m3", "f8e5m2"};
    return valid_types;
}

// struct Evaluate : element::NoAction<bool> {
//     using element::NoAction<bool>::visit;
// };

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

// bool FakeConvert::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
//     OV_OP_SCOPE(v13_FakeConvert_evaluate);
//     OPENVINO_ASSERT(outputs.size() == 1);
//     OPENVINO_ASSERT(inputs.size() == 3);

//     const auto& data_shape = inputs[0].get_shape();
//     outputs[0].set_shape(data_shape);

//     return true;
// }

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


namespace convert_fp8 {
namespace {
void print_tensor(const ov::Tensor& t, std::string s) {
    std::cout << "Tensor " << s << ": ";
    auto shape = t.get_shape();
    int len = shape_size(shape);

    if (t.get_element_type() == ov::element::f16) {
        auto ptr = static_cast<ov::float16*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    if (t.get_element_type() == ov::element::f32) {
        auto ptr = static_cast<float*>(t.data());
        for (int i = 0; i < len; i++) {
            std::cout << ptr[i] << " ";
        }
    }
    std::cout << std::endl;
}


template <typename ET>
bool evaluate(ov::Tensor& arg, ov::Tensor& out, const std::string& destination_type) {
    out.set_shape(arg.get_shape());
    size_t element_count = shape_size(out.get_shape());

    if ((ov::element::f16 != arg.get_element_type()) || ov::element::f16 != out.get_element_type()) {
        std::cout << "Bad arg or out types: " << arg.get_element_type() << " " << out.get_element_type() << std::endl;
        return false;
    }

    auto inPtr = static_cast<ET*>(arg.data());
    auto outPtr = static_cast<ET*>(out.data());

    if (destination_type == "f8e5m2") {
        reference::fake_convert::convertfp16_bf8(inPtr, outPtr, element_count);
    } else if (destination_type == "f8e4m3") {
        reference::fake_convert::convertfp16_f8e4m3_bias7(inPtr, outPtr, element_count);
    } else {
        std::cout << "Bad destination_type: " << destination_type << std::endl;
    }

    return true;
}
}  // namespace
}  // namespace convert_fp8



bool FakeConvert::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    ov::TensorVector fp16;

    OPENVINO_ASSERT(
        (outputs[0].get_element_type() == ov::element::f32 && inputs[0].get_element_type() == ov::element::f32) ||
            (outputs[0].get_element_type() == ov::element::f16 && inputs[0].get_element_type() == ov::element::f16),
        "Wrong input or output type for FakeConvertFP::evaluate");

    outputs[0].set_shape(inputs[0].get_shape());
    const auto element_count = inputs[0].get_size();

    if (inputs[0].get_element_type() == ov::element::f16) {
        fp16.emplace_back(inputs[0]);
        if (m_apply_scale) {
            reference::fake_convert::apply_per_channel_scale<float16>(fp16[0], inputs[1], inputs[2]);
        }
        convert_fp8::evaluate<unsigned short>(fp16[0], outputs[0], m_destination_type);
        if (m_apply_scale) {
            reference::fake_convert::apply_per_channel_scale<float16>(outputs[0], inputs[1], inputs[2], true);
        }
    } else if (inputs[0].get_element_type() == ov::element::f32) {
        fp16.emplace_back(ov::Tensor(ov::element::f16, inputs[0].get_shape()));
        if (m_apply_scale) {
            auto data_fp32 = ov::Tensor(ov::element::f32, inputs[0].get_shape());
            inputs[0].copy_to(data_fp32);
            reference::fake_convert::apply_per_channel_scale<float>(data_fp32, inputs[1], inputs[2]);
            ov::reference::convert(data_fp32.data<float>(), fp16[0].data<ov::float16>(), element_count);
        } else {
            ov::reference::convert(inputs[0].data<float>(), fp16[0].data<ov::float16>(), element_count);
        }
        convert_fp8::evaluate<unsigned short>(fp16[0], fp16[0], m_destination_type);
        ov::reference::convert(fp16[0].data<ov::float16>(), outputs[0].data<float>(), element_count);
        if (m_apply_scale) {
            reference::fake_convert::apply_per_channel_scale<float>(outputs[0], inputs[1], inputs[2], true);
        }
    }

    return true;
}

}  // namespace v13
}  // namespace op
}  // namespace ov
