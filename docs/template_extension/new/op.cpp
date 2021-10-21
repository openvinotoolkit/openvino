// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op.hpp"

using namespace TemplateExtension;

//! [op:ctor]
Operation::Operation(const ov::Output<ov::Node>& arg, int64_t add) : Op({arg}), add(add) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void Operation::validate_and_infer_types() {
    // Operation doesn't change shapes end element type
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> Operation::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() != 1,"Incorrect number of new arguments")

    return std::make_shared<Operation>(new_args.at(0), add);
}
//! [op:copy]

//! [op:visit_attributes]
bool Operation::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("add", add);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
namespace {

template <class T>
void implementation(const T* input, T* output, int64_t add, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = input[i] + add;
    }
}

template <ov::element::Type_t ET>
bool evaluate_op(const ov::runtime::Tensor& arg0, ov::runtime::Tensor& out, int64_t add) {
    size_t size = ov::shape_size(arg0.get_shape());
    implementation(arg0.data<typename ov::element_type_traits<ET>::value_type>(),
                   out.data<typename ov::element_type_traits<ET>::value_type>(),
                   add,
                   size);
    return true;
}

}  // namespace

bool Operation::evaluate(ov::runtime::TensorVector& outputs, const ov::runtime::TensorVector& inputs) const {
    switch (inputs[0].get_element_type()) {
    case ov::element::Type_t::i8:
        return evaluate_op<ov::element::Type_t::i8>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::i16:
        return evaluate_op<ov::element::Type_t::i16>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::i32:
        return evaluate_op<ov::element::Type_t::i32>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::i64:
        return evaluate_op<ov::element::Type_t::i64>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::u8:
        return evaluate_op<ov::element::Type_t::u8>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::u16:
        return evaluate_op<ov::element::Type_t::u16>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::u32:
        return evaluate_op<ov::element::Type_t::u32>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::u64:
        return evaluate_op<ov::element::Type_t::u8>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::bf16:
        return evaluate_op<ov::element::Type_t::bf16>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::f16:
        return evaluate_op<ov::element::Type_t::f16>(inputs[0], outputs[0], getAddAttr());
    case ov::element::Type_t::f32:
        return evaluate_op<ov::element::Type_t::f32>(inputs[0], outputs[0], getAddAttr());
    default:
        break;
    }
    return false;
}

bool Operation::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ov::element::Type_t::i8:
    case ov::element::Type_t::i16:
    case ov::element::Type_t::i32:
    case ov::element::Type_t::i64:
    case ov::element::Type_t::u8:
    case ov::element::Type_t::u16:
    case ov::element::Type_t::u32:
    case ov::element::Type_t::u64:
    case ov::element::Type_t::bf16:
    case ov::element::Type_t::f16:
    case ov::element::Type_t::f32:
        return true;
    default:
        break;
    }
    return false;
}
//! [op:evaluate]
