// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mish.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/mish.hpp"

namespace ov {
namespace op {
namespace mish {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        reference::mish(arg.data<const T>(), out.data<T>(), count);
        return true;
    }
};
}  // namespace mish

namespace v4 {

Mish::Mish(const Output<Node>& arg) : util::UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void Mish::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Mish_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Only accepts one argument. Got: ", get_input_size());

    const auto& data_batch_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_batch_et.is_real() || data_batch_et.is_dynamic(),
                          "Element must be of floating point type, Got: ",
                          data_batch_et);

    set_output_type(0, data_batch_et, get_input_partial_shape(0));
}

std::shared_ptr<Node> Mish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Mish_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Mish>(new_args.at(0));
}

bool Mish::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_Mish_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& in_shape = inputs[0].get_shape();
    outputs[0].set_shape(in_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_Mish_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      mish::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(in_shape));
}

bool Mish::has_evaluate() const {
    OV_OP_SCOPE(v4_Mish_has_evaluate);
    switch (get_input_element_type(0)) {
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
