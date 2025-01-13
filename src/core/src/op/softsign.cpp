// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/softsign.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/reference/softsign.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace softsign {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        reference::softsign(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};

}  // namespace softsign
namespace v9 {
SoftSign::SoftSign(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void SoftSign::validate_and_infer_types() {
    OV_OP_SCOPE(v9_SoftSign_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float, instead got: ",
                          input_et);

    UnaryElementwiseArithmetic::validate_and_infer_types();
}

std::shared_ptr<Node> SoftSign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_SoftSign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SoftSign>(new_args.at(0));
}

bool SoftSign::has_evaluate() const {
    OV_OP_SCOPE(v9_SoftSign_has_evaluate);
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

bool SoftSign::evaluate(TensorVector& outputs,
                        const TensorVector& inputs,
                        const EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v9_SoftSign_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1 && inputs.size() == 1,
                    "SoftSign evaluate needs exactly 1 input and 1 output, instead got:",
                    inputs.size(),
                    " input(s) and ",
                    outputs.size(),
                    " output(s).");

    const auto& input_shape = inputs[0].get_shape();
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v9_SoftSign_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      softsign::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(input_shape));
}
}  // namespace v9
}  // namespace op
}  // namespace ov
