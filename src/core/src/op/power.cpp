// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/power.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/power.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace power {
struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0,
                             const Tensor& arg1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& b_spec) {
        reference::power(arg0.data<const T>(), arg1.data<const T>(), out.data<T>(), shape0, shape1, b_spec);
        return true;
    }
};
}  // namespace power

// ------------------------------ v1 -------------------------------------------
namespace v1 {
Power::Power(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Power::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Power_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Power>(new_args.at(0), new_args.at(1), get_autob());
}

bool Power::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Power_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    auto& out = outputs[0];
    out.set_shape(infer_broadcast_shape(this, inputs));

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Power_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      power::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      out,
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool Power::has_evaluate() const {
    OV_OP_SCOPE(v1_Power_has_evaluate);
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
}  // namespace v1
}  // namespace op
}  // namespace ov
