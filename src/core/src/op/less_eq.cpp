// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/less_eq.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/less_eq.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace less_equal {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        reference::less_eq(in0.data<const T>(),
                           in1.data<const T>(),
                           out.data<fundamental_type_for<element::boolean>>(),
                           shape0,
                           shape1,
                           broadcast_spec);
        return true;
    }
};
}  // namespace less_equal

// ----------------------------- v1 --------------------------------------------
namespace v1 {
LessEqual::LessEqual(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LessEqual::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LessEqual_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LessEqual>(new_args.at(0), new_args.at(1), get_autob());
}

bool LessEqual::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_LessEqual_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_LessEqual_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, i32, i64, u32, u64),
                                      less_equal::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool LessEqual::has_evaluate() const {
    OV_OP_SCOPE(v1_LessEqual_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::boolean:
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
