// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/hswish.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/hswish.hpp"

namespace ov {
namespace op {
namespace v4 {
HSwish::HSwish(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> HSwish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_HSwish_clone_with_new_inputs);
    return std::make_shared<HSwish>(new_args.at(0));
}

namespace hswish {
namespace {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        ov::reference::hswish(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace
}  // namespace hswish

bool HSwish::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_HSwish_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    const auto count = shape_size(input_shape);
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_HSwish_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      hswish::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      count);
}

bool HSwish::has_evaluate() const {
    OV_OP_SCOPE(v4_HSwish_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v4
}  // namespace op
}  // namespace ov
