// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/prelu.hpp"

namespace ov {
namespace op {
namespace prelu {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg,
                             const Tensor& slope,
                             Tensor& out,
                             const Shape& arg_shape,
                             const Shape& slope_shape) {
        reference::prelu(arg.data<const T>(), slope.data<const T>(), out.data<T>(), arg_shape, slope_shape);
        return true;
    }
};
}  // namespace prelu

namespace v0 {

PRelu::PRelu() : Op() {}

PRelu::PRelu(const Output<Node>& data, const Output<Node>& slope) : Op({data, slope}) {
    constructor_validate_and_infer_types();
}

void PRelu::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> PRelu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_PRelu_clone_with_new_inputs);
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<PRelu>(new_args.at(0), new_args.at(1));
}

bool PRelu::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_PRelu_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    auto& out = outputs[0];
    const auto& arg_shape = inputs[0].get_shape();
    out.set_shape(arg_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_PRelu_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8),
                                      prelu::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      out,
                                      arg_shape,
                                      inputs[1].get_shape());
}

bool PRelu::has_evaluate() const {
    OV_OP_SCOPE(v0_PRelu_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i8:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
