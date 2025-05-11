// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cosh.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/cosh.hpp"

using namespace std;

namespace ov {
namespace op {
namespace cosh {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::cosh(arg0.data<T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace cosh

namespace v0 {
Cosh::Cosh(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool Cosh::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Cosh_visit_attributes);
    return true;
}

shared_ptr<Node> Cosh::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Cosh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Cosh>(new_args.at(0));
}

bool Cosh::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Cosh_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Cosh_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      cosh::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Cosh::has_evaluate() const {
    OV_OP_SCOPE(v0_Cosh_has_evaluate);
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
