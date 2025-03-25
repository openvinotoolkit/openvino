// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sqrt.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/sqrt.hpp"

namespace ov {
namespace op {
namespace sqrt {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count) {
        reference::sqrt(arg0.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace sqrt

namespace v0 {
Sqrt::Sqrt(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Sqrt::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Sqrt_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Sqrt>(new_args.at(0));
}

bool Sqrt::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Sqrt_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Sqrt_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64, i32, i64, u32, u64),
                                      sqrt::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(in_shape));
}

bool Sqrt::has_evaluate() const {
    OV_OP_SCOPE(v0_Sqrt_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
    case element::f64:
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
