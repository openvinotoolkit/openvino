// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/logical_not.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/logical_not.hpp"

namespace ov {
namespace op {
namespace logical_not {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        reference::logical_not(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace logical_not

namespace v1 {

LogicalNot::LogicalNot(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void LogicalNot::validate_and_infer_types() {
    OV_OP_SCOPE(v1_LogicalNot_validate_and_infer_types);
    const auto& element_type = get_input_element_type(0);
    // No boolean element_type validation for backward compatibility
    const auto& arg_pshape = get_input_partial_shape(0);
    set_output_type(0, element_type, arg_pshape);
}

std::shared_ptr<Node> LogicalNot::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LogicalNot_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<LogicalNot>(new_args.at(0));
}

bool LogicalNot::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_LogicalNot_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_LogicalNot_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, u8, i32, i64, u32, u64, f32),
                                      logical_not::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool LogicalNot::has_evaluate() const {
    OV_OP_SCOPE(v1_LogicalNot_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::boolean:
    case element::u8:
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
