// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/bitwise_right_shift.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/elementwise_args.hpp"
#include "openvino/reference/bitwise_right_shift.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
namespace right_shift {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& arg0_shape,
                             const Shape& arg1_shape,
                             const op::AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::bitwise_right_shift(in0.data<const T>(),
                                       in1.data<const T>(),
                                       out.data<T>(),
                                       arg0_shape,
                                       arg1_shape,
                                       broadcast_spec);
        return true;
    }
};
}  // namespace right_shift

BitwiseRightShift::BitwiseRightShift(const Output<Node>& arg0,
                                     const Output<Node>& arg1,
                                     const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseBitwise(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> BitwiseRightShift::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_BitwiseRightShift_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BitwiseRightShift>(new_args[0], new_args[1], get_autob());
}

bool BitwiseRightShift::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_BitwiseRightShift_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF(v15_BitwiseRightShift_evaluate,
                      OV_PP_ET_LIST(i32),
                      right_shift::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      inputs[1],
                      outputs[0],
                      inputs[0].get_shape(),
                      inputs[1].get_shape(),
                      get_autob());
}

void BitwiseRightShift::validate_and_infer_types() {
    OV_OP_SCOPE(v15_BitwiseRightShift_validate_and_infer_types);
    auto args_et_pshape = op::util::validate_and_infer_elementwise_args(this);
    const auto& args_et = std::get<0>(args_et_pshape);
    const auto& args_pshape = std::get<1>(args_et_pshape);

    NODE_VALIDATION_CHECK(this,
                          args_et.is_dynamic() || args_et.is_integral_number(),
                          "The element type of the input tensor must be integer number.");

    set_output_type(0, args_et, args_pshape);
}

bool BitwiseRightShift::has_evaluate() const {
    OV_OP_SCOPE(v15_BitwiseRightShift_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
        return true;
    default:
        return false;
    }
}
}  // namespace v15
}  // namespace op
}  // namespace ov
