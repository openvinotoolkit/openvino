// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/softplus.hpp"

namespace ov {
namespace op {
namespace v4 {
SoftPlus::SoftPlus(const Output<Node>& arg) : util::UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void SoftPlus::validate_and_infer_types() {
    OV_OP_SCOPE(v4_SoftPlus_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float. Got: ",
                          input_et);

    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> SoftPlus::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_SoftPlus_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SoftPlus>(new_args.at(0));
}

namespace softplus {
namespace {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        ov::reference::softplus(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace
}  // namespace softplus

bool SoftPlus::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_SoftPlus_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    const auto count = shape_size(input_shape);
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_SoftPlus_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      softplus::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      count);
}

bool SoftPlus::has_evaluate() const {
    OV_OP_SCOPE(v4_SoftPlus_has_evaluate);
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
