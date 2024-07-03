// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/subtract.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace subtract {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const AutoBroadcastSpec& broadcast_spec) {
        using T = typename element_type_traits<ET>::value_type;
        reference::subtract(in0.data<const T>(), in1.data<const T>(), out.data<T>(), shape0, shape1, broadcast_spec);
        return true;
    }
};
}  // namespace subtract

// ------------------------------- v1 ------------------------------------------
namespace v1 {
Subtract::Subtract(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Subtract::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Subtract_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Subtract>(new_args.at(0), new_args.at(1), get_autob());
}

bool Subtract::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Subtract_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Subtract_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i8, i32, i64, u8, u32, u64),
                                      subtract::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob());
}

bool Subtract::evaluate_symbol(ov::TensorSymbolVector& output_symbols) const {
    auto lhs_pshape = input(0).get_tensor().get_partial_shape(), rhs_pshape = input(1).get_tensor().get_partial_shape();
    auto lhs_symbols = input(0).get_tensor().get_value_symbol(), rhs_symbols = input(1).get_tensor().get_value_symbol();
    if (lhs_pshape.is_dynamic() || rhs_pshape.is_dynamic() || lhs_pshape != rhs_pshape || lhs_symbols.empty() ||
        lhs_symbols.size() != rhs_symbols.size())
        return false;  // broadcasting is not supported here yet
    output_symbols.resize(1);
    output_symbols[0].resize(shape_size(lhs_pshape.to_shape()));
    for (size_t i = 0; i < output_symbols[0].size(); ++i)
        output_symbols[0][i] = lhs_symbols[i] - rhs_symbols[i];
    return true;
}

bool Subtract::has_evaluate() const {
    OV_OP_SCOPE(v1_Subtract_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::i8:
    case element::i32:
    case element::i64:
    case element::u8:
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
