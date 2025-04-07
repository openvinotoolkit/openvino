// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tan.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/tan.hpp"

namespace ov {
namespace op {
namespace tan {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::tan(arg0.data<T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace tan

namespace v0 {
Tan::Tan(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool Tan::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Tan_visit_attributes);
    return true;
}

std::shared_ptr<Node> Tan::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Tan_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Tan>(new_args.at(0));
}

bool Tan::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Tan_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Tan_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      tan::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Tan::has_evaluate() const {
    OV_OP_SCOPE(v0_Tan_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
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
