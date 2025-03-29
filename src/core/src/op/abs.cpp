// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/tensor_util.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/reference/abs.hpp"

namespace ov {
namespace op {
namespace abs {
struct Evaluate : ov::element::NoAction<bool> {
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in, Tensor& out, const size_t count) {
        using T = typename element_type_traits<ET>::value_type;
        reference::abs(in.data<const T>(), out.data<T>(), count);
        return true;
    }
};

static bool evaluate_bound(const ov::op::v0::Abs* op, TensorVector& output_values, bool is_lower) {
    const auto &lv = op->get_input_tensor(0).get_lower_value(), &uv = op->get_input_tensor(0).get_upper_value();
    if (!lv || !uv || !op->has_evaluate())
        return false;
    TensorVector lower_output{ov::Tensor(lv.get_element_type(), lv.get_shape())},
        upper_output{ov::Tensor(uv.get_element_type(), uv.get_shape())};
    if (!op->evaluate(lower_output, {lv}) || !op->evaluate(upper_output, {uv}))
        return false;
    if (is_lower)
        return v1::Minimum().evaluate(output_values, {lower_output[0], upper_output[0]});
    else
        return v1::Maximum().evaluate(output_values, {lower_output[0], upper_output[0]});
}
}  // namespace abs

namespace v0 {
Abs::Abs(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Abs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Abs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Abs>(new_args.at(0));
}

bool Abs::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Abs_evaluate);

    OPENVINO_ASSERT(inputs.size() == 1);
    OPENVINO_ASSERT(outputs.size() == 1);
    outputs[0].set_shape(inputs[0].get_shape());

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v0_Abs_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      abs::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      shape_size(inputs[0].get_shape()));
}

bool Abs::has_evaluate() const {
    OV_OP_SCOPE(v0_Abs_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
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

bool Abs::evaluate_lower(TensorVector& output_values) const {
    return ov::op::abs::evaluate_bound(this, output_values, true);
}

bool Abs::evaluate_upper(TensorVector& output_values) const {
    return ov::op::abs::evaluate_bound(this, output_values, false);
}

bool Abs::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    const auto& non_negative = ov::util::greater_equal(get_input_tensor(0).get_lower_value(), 0);
    const auto& input_symbols = get_input_tensor(0).get_value_symbol();
    if (!non_negative || input_symbols.size() != ov::shape_size(non_negative.get_shape()))
        return false;
    auto non_negative_ptr = non_negative.data<char>();
    output_symbols.resize(1);
    output_symbols[0].resize(input_symbols.size(), nullptr);
    for (size_t i = 0; i < input_symbols.size(); ++i)
        if (non_negative_ptr[i])
            output_symbols[0][i] = input_symbols[i];
    return true;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
