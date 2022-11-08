// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op.hpp"

using namespace TemplateExtension;

//! [op:ctor]
Operation::Operation(const ngraph::Output<ngraph::Node>& arg, int64_t add) : Op({arg}), add(add) {
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
std::shared_ptr<ngraph::Node> Operation::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }

    return std::make_shared<Operation>(new_args.at(0), add);
}
//! [op:copy]

//! [op:visit_attributes]
bool Operation::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("add", add);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
namespace {

template <class T>
void implementation(const T* input, T* output, int64_t add, size_t size) {
    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<T>(input[i] + add);
    }
}

template <ngraph::element::Type_t ET>
bool evaluate_op(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out, int64_t add) {
    size_t size = ngraph::shape_size(arg0->get_shape());
    implementation(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), add, size);
    return true;
}

}  // namespace

bool Operation::evaluate(const ngraph::HostTensorVector& outputs, const ngraph::HostTensorVector& inputs) const {
    switch (inputs[0]->get_element_type()) {
    case ngraph::element::Type_t::i8:
        return evaluate_op<ngraph::element::Type_t::i8>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::i16:
        return evaluate_op<ngraph::element::Type_t::i16>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::i32:
        return evaluate_op<ngraph::element::Type_t::i32>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::i64:
        return evaluate_op<ngraph::element::Type_t::i64>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::u8:
        return evaluate_op<ngraph::element::Type_t::u8>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::u16:
        return evaluate_op<ngraph::element::Type_t::u16>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::u32:
        return evaluate_op<ngraph::element::Type_t::u32>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::u64:
        return evaluate_op<ngraph::element::Type_t::u8>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::bf16:
        return evaluate_op<ngraph::element::Type_t::bf16>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::f16:
        return evaluate_op<ngraph::element::Type_t::f16>(inputs[0], outputs[0], getAddAttr());
    case ngraph::element::Type_t::f32:
        return evaluate_op<ngraph::element::Type_t::f32>(inputs[0], outputs[0], getAddAttr());
    default:
        break;
    }
    return false;
}

bool Operation::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ngraph::element::Type_t::i8:
    case ngraph::element::Type_t::i16:
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::i64:
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::u16:
    case ngraph::element::Type_t::u32:
    case ngraph::element::Type_t::u64:
    case ngraph::element::Type_t::bf16:
    case ngraph::element::Type_t::f16:
    case ngraph::element::Type_t::f32:
        return true;
    default:
        break;
    }
    return false;
}
//! [op:evaluate]
