// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/add.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace add {
struct Evaluate : element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::add(in0.data<const T>(), in1.data<const T>(), out.data<T>(), shape0, shape1, broadcast_spec);
        return true;
    }
};
}  // namespace add

// ------------------------------- v1 ------------------------------------------
namespace v1 {
Add::Add(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Add::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Add_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v1::Add>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool Add::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Add_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Add_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i16, i32, i64, u8, u16, u32, u64),
                                      add::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool Add::has_evaluate() const {
    OV_OP_SCOPE(v1_Add_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
