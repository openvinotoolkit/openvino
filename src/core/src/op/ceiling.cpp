// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ceiling.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/ceiling.hpp"

namespace ov {
namespace op {
namespace ceiling {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::ceiling(arg0.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace ceiling

namespace v0 {
Ceiling::Ceiling(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Ceiling::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Ceiling_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Ceiling>(new_args.at(0));
}

bool Ceiling::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Ceiling_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF(v0_Ceiling_evaluate,
                      OV_PP_ET_LIST(f16, f32, i8, i16, i32, i64, u8, u16, u32, u64),
                      ceiling::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      outputs[0],
                      shape_size(inputs[0].get_shape()));
}

bool Ceiling::has_evaluate() const {
    OV_OP_SCOPE(v0_Ceiling_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
