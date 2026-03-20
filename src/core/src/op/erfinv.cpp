// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/erfinv.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/reference/erfinv.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {

namespace erfinv {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        reference::erfinv(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace erfinv

namespace v16 {

ErfInv::ErfInv(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void ErfInv::validate_and_infer_types() {
    OV_OP_SCOPE(v16_ErfInv_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be floating-point, instead got: ",
                          input_et);

    UnaryElementwiseArithmetic::validate_and_infer_types();
}

std::shared_ptr<Node> ErfInv::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v16_ErfInv_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ErfInv>(new_args.at(0));
}

bool ErfInv::has_evaluate() const {
    OV_OP_SCOPE(v16_ErfInv_has_evaluate);
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

bool ErfInv::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v16_ErfInv_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1 && inputs.size() == 1,
                    "ErfInv evaluate needs exactly 1 input and 1 output, instead got: ",
                    inputs.size(),
                    " input(s) and ",
                    outputs.size(),
                    " output(s).");

    const auto& input_shape = inputs[0].get_shape();
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v16_ErfInv_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      erfinv::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(input_shape));
}

}  // namespace v16
}  // namespace op
}  // namespace ov
