// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"
#include "openvino/reference/bitwise_left_shift.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
namespace left_shift {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0, const Tensor& in1, Tensor& out) {
        using T = typename element_type_traits<ET>::value_type;
        reference::bitwise_left_shift(in0.data<const T>(),
                                      in1.data<const T>(),
                                      out.data<T>(),
                                      in0.get_shape(),
                                      in1.get_shape());
        return true;
    }
};

namespace {
bool evaluate(TensorVector& outputs, const TensorVector& inputs) {
    using namespace ov::element;
    return IF_TYPE_OF(bitshift_evaluate,
                      OV_PP_ET_LIST(i8, i16, i32, i64, u8, u16, u32, u64),
                      left_shift::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      inputs[1],
                      outputs[0]);
}
}  // namespace
}  // namespace left_shift

BitwiseLeftShift::BitwiseLeftShift(const Output<Node>& arg0,
                                   const Output<Node>& arg1,
                                   const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseBitwise(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> BitwiseLeftShift::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_BitwiseLeftShift_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BitwiseLeftShift>(new_args[0], new_args[1], get_autob());
}

bool BitwiseLeftShift::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_BitwiseLeftShift_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 2);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    return left_shift::evaluate(outputs, inputs);
}

bool BitwiseLeftShift::has_evaluate() const {
    OV_OP_SCOPE(v15_BitwiseLeftShift_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
    return false;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
