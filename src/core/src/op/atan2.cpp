// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/atan2.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/atan2.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace atan2 {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in_y,
                             const Tensor& in_x,
                             Tensor& out,
                             const Shape& y_shape,
                             const Shape& x_shape,
                             const AutoBroadcastSpec& broadcast_spec) {
        reference::atan2(in_y.data<const T>(), in_x.data<const T>(), out.data<T>(), y_shape, x_shape, broadcast_spec);
        return true;
    }
};
}  // namespace atan2

namespace v17 {
Atan2::Atan2(const Output<Node>& y, const Output<Node>& x, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(y, x, auto_broadcast) {
    constructor_validate_and_infer_types();
}

void Atan2::validate_and_infer_types() {
    OV_OP_SCOPE(v17_Atan2_validate_and_infer_types);
    BinaryElementwiseArithmetic::validate_and_infer_types();
    // BinaryElementwiseArithmetic ensures both inputs have the same type,
    // so checking input 0 is sufficient to validate both.
    // Allow dynamic type to support partial inference during graph construction.
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
                          "Atan2 inputs must be floating-point type, got: ",
                          get_input_element_type(0));
}

std::shared_ptr<Node> Atan2::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_Atan2_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v17::Atan2>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool Atan2::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_Atan2_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v17_Atan2_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      atan2::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool Atan2::has_evaluate() const {
    OV_OP_SCOPE(v17_Atan2_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v17
}  // namespace op
}  // namespace ov
