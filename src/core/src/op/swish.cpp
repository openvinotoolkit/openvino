// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/swish.hpp"

#include "compare.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/reference/swish.hpp"

namespace ov {
namespace op {
namespace swish {
constexpr auto has_1_or_2_inputs = ov::cmp::Between<size_t>(0, 3);

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& arg0, const Tensor& arg1, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::swish(arg0.data<const T>(), arg1 ? *arg1.data<const T>() : T{1.0}, out.data<T>(), count);
        return true;
    }
};
}  // namespace swish

namespace v4 {
Swish::Swish(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

Swish::Swish(const Output<Node>& arg, const Output<Node>& beta) : Op({arg, beta}) {
    constructor_validate_and_infer_types();
}

void Swish::validate_and_infer_types() {
    OV_OP_SCOPE(v4_Swish_validate_and_infer_types);

    const auto inputs_count = input_values().size();
    NODE_VALIDATION_CHECK(this,
                          swish::has_1_or_2_inputs(inputs_count),
                          "Swish must have 1 or 2 inputs, but it has: ",
                          inputs_count);

    auto in_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          in_type.is_dynamic() || in_type.is_real(),
                          "Swish input tensor must be floating point type(",
                          in_type,
                          ").");

    if (inputs_count == 2) {
        NODE_VALIDATION_CHECK(this,
                              input_value(0).get_element_type() == input_value(1).get_element_type(),
                              "Swish inputs must have the same type but they are: ",
                              input_value(0).get_element_type(),
                              " and ",
                              input_value(1).get_element_type());

        const auto beta_rank = get_input_partial_shape(1).rank();
        NODE_VALIDATION_CHECK(this,
                              beta_rank.compatible(0),
                              "Swish input with beta must be scalar but it has rank: ",
                              beta_rank);
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> Swish::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_Swish_clone_with_new_inputs);
    if (new_args.size() == 1) {
        return std::make_shared<Swish>(new_args.at(0));
    } else {
        return std::make_shared<Swish>(new_args.at(0), new_args.at(1));
    }
}

bool Swish::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v4_Swish_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(swish::has_1_or_2_inputs(inputs.size()));

    outputs[0].set_shape(inputs[0].get_shape());
    const auto& arg1 = inputs.size() == 2 ? inputs[1] : Tensor();

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v4_Swish_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      swish::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      arg1,
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Swish::has_evaluate() const {
    OV_OP_SCOPE(v4_Swish_has_evaluate);
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
