// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/abs.hpp"

namespace ov {
namespace op {
namespace abs {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::abs(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace abs

namespace v0 {
Abs::Abs(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Abs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Abs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Abs>(new_args.at(0));
}

bool Abs::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Abs_evaluate);

    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IfTypeOf<bf16, f16, f32, i32, i64, u32, u64>::apply<abs::Evaluate>(inputs[0].get_element_type(),
                                                                              inputs[0],
                                                                              outputs[0],
                                                                              shape_size(inputs[0].get_shape()));
}

bool Abs::has_evaluate() const {
    OV_OP_SCOPE(v0_Abs_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
