// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/negative.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/negate.hpp"

namespace ov {
namespace op {
namespace negative {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count) {
        reference::negate(arg0.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace negative

namespace v0 {

Negative::Negative(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Negative::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Negative_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Negative>(new_args.at(0));
}

bool Negative::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Negative_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    outputs[0].set_shape(inputs[0].get_shape());
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Negative_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64),
                                      negative::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Negative::has_evaluate() const {
    OV_OP_SCOPE(v0_Negative_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
