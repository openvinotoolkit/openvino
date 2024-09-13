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
    constructor_validate_and_infer_types();
}

void DecodeImg::validate_and_infer_types() {
    OV_OP_SCOPE(v0_DecodeImg_validate_and_infer_types);
    ov::PartialShape output_shape{Dimension(), Dimension(),3};
    set_output_type(0, ov::element::u8, output_shape);
}

bool DecodeImg::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_DecodeImg_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> DecodeImg::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_DecodeImg_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto num_args = new_args.size();
    NODE_VALIDATION_CHECK(this, num_args == 1, "DecodeImg accepts 1 inputs.");
    return std::make_shared<DecodeImg>(new_args.at(0));
}

bool DecodeImg::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_DecodeImg_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    if (ov::reference::decodeimg(inputs[0], outputs[0])) {
        ov::PartialShape output_shape{Dimension(), Dimension(),3};
        outputs[0].set_shape(output_shape.to_shape());
    }
    return true;
}

bool DecodeImg::has_evaluate() const {
    OV_OP_SCOPE(v0_DecodeImg_has_evaluate);
    const auto& input_0_et = get_input_element_type(0);
    return input_0_et == element::u8 || input_0_et == element::i8;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
