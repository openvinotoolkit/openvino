// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/decodeimg.hpp"
#include "openvino/reference/decodeimg.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"

namespace ov {
namespace op {
namespace v0 {
// ------------------------------ V0 ------------------------------
DecodeImg::DecodeImg(const Output<Node>& arg) 
 : Op({arg}) {
    std::cout << "in DecodeImg::DecodeImg" << std::endl;
    std::cout.flush();
    m_width = 224;
    m_height = 224;
    m_channel = 3;
    constructor_validate_and_infer_types();
}

void DecodeImg::validate_and_infer_types() {
    std::cout << "in DecodeImg::validate_and_infer_types" << std::endl;
    std::cout.flush();
    OV_OP_SCOPE(v0_DecodeImg_validate_and_infer_types);
    // ov::PartialShape output_shape{2,Dimension::dynamic(),3};
    ov::PartialShape output_shape{224, 224, 3};
    set_output_type(0, ov::element::i32, output_shape);
}

bool DecodeImg::visit_attributes(AttributeVisitor& visitor) {
    std::cout << "in DecodeImg::visit_attributes " << std::endl;
    std::cout.flush();
    OV_OP_SCOPE(v0_DecodeImg_visit_attributes);
    // visitor.on_attribute("num_classes", m_attrs.num_classes);
    // visit_attributes_base(visitor, m_attrs);
    return true;
}

std::shared_ptr<ov::Node> DecodeImg::clone_with_new_inputs(const OutputVector& new_args) const {
    std::cout << "in DecodeImg::clone_with_new_inputs new_args.size=" << new_args.size() << std::endl;
    std::cout.flush();
    OV_OP_SCOPE(v0_DecodeImg_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    auto num_args = new_args.size();

    NODE_VALIDATION_CHECK(this, num_args == 1, "DecodeImg accepts 1 inputs.");

    return std::make_shared<DecodeImg>(new_args.at(0));
}

bool DecodeImg::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_DecodeImg_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);

    // const auto out_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    // const ov::Shape out_shape = std::vector<uint64_t>({244,244,3});

    // outputs[0].set_shape(out_shape);
    // std::cout << "in DecodeImg::evaluate. inputs.size() =" << inputs.size() 
    // << ", outputs.size()=" << outputs.size() << std::endl;

    ov::reference::decodeimg(inputs[0], outputs[0]);

    return true;
}

bool DecodeImg::has_evaluate() const {
    OV_OP_SCOPE(v0_DecodeImg_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    std::cout << "in DecodeImg::has_evaluate input_0_et=" << input_0_et << std::endl;
    return true;//input_0_et == element::string;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
